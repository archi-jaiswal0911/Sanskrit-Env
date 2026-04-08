"""
Submission-safe inference script for SanskritEnv.

This script is intentionally narrow:
- Runs a single episode for a single task.
- Emits only [START], [STEP], and [END] lines to stdout.
- Sends diagnostics to stderr.
- Handles network, parsing, and environment failures without crashing.
"""

import json
import logging
import os
import re
import sys
import time
from contextlib import redirect_stderr
from io import StringIO
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from client import SanskritEnv
from models import ManuscriptAction


logging.getLogger("dotenv.main").setLevel(logging.ERROR)

with redirect_stderr(StringIO()):
    load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _first_nonempty_env(*names: str) -> Tuple[str, Optional[str]]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            normalized = value.strip().strip('"').strip("'")
            if normalized.lower().startswith("bearer "):
                normalized = normalized[7:].strip()
            return normalized, name
    return "", None


HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HF_API_TOKEN",
    "HF_API_KEY",
    "HUGGINGFACE_TOKEN",
    "API_KEY",
)

CLOUDFLARE_TOKEN_ENV_KEYS = (
    "CLOUDFLARE_API_TOKEN",
    "CF_API_TOKEN",
    "CLOUDFLARE_TOKEN",
    "CF_TOKEN",
)

HF_TOKEN, _ = _first_nonempty_env(*HF_TOKEN_ENV_KEYS)
CLOUDFLARE_API_TOKEN, _ = _first_nonempty_env(*CLOUDFLARE_TOKEN_ENV_KEYS)
CLOUDFLARE_ACCOUNT_ID, _ = _first_nonempty_env("CLOUDFLARE_ACCOUNT_ID", "CF_ACCOUNT_ID")

HF_ROUTER_URL = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")

DEFAULT_CF_CHAT_URL = ""
if CLOUDFLARE_ACCOUNT_ID:
    DEFAULT_CF_CHAT_URL = (
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
    )

DEFAULT_HF_CHAT_URL = HF_ROUTER_URL

MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    or os.environ.get("BASELINE_MODEL")
    or "Qwen/Qwen2.5-72B-Instruct"
).strip()


def _default_api_base_url(model_name: str) -> str:
    normalized_model = (model_name or "").strip().lower()

    if normalized_model.startswith("@cf/"):
        return DEFAULT_CF_CHAT_URL or DEFAULT_HF_CHAT_URL

    return DEFAULT_HF_CHAT_URL or DEFAULT_CF_CHAT_URL

API_BASE_URL = (
    os.environ.get("API_BASE_URL")
    or os.environ.get("CF_CHAT_COMPLETIONS_URL")
    or os.environ.get("CF_WORKERS_AI_URL")
    or _default_api_base_url(MODEL_NAME)
).strip()

ENV_URL = (
    os.environ.get("SANSKRIT_ENV_URL")
    or os.environ.get("ENV_URL")
    or os.environ.get("OPENENV_BASE_URL")
    or "http://127.0.0.1:7860"
).rstrip("/")

TASK_NAME = (
    os.environ.get("TASK_NAME")
    or os.environ.get("OPENENV_TASK")
    or os.environ.get("BASELINE_TASK")
    or "glossary_anchoring"
).strip().lower()

RUN_CURRICULUM = TASK_NAME == "all"

VALID_TASKS = {
    "glossary_anchoring",
    "sandhi_resolution",
    "samasa_classification",
    "referential_coherence",
}
TASK_SEQUENCE = [
    "glossary_anchoring",
    "sandhi_resolution",
    "samasa_classification",
    "referential_coherence",
]

if not RUN_CURRICULUM and TASK_NAME not in VALID_TASKS:
    TASK_NAME = "glossary_anchoring"

BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "sanskrit-env")
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
MAX_STEPS = _env_int("MAX_STEPS", 8)
TASK_MAX_STEPS = {
    "glossary_anchoring": 1,
    "sandhi_resolution": 1,
    "samasa_classification": 1,
    "referential_coherence": 7,
}
TARGET_STEPS_PER_TASK = _env_int("TARGET_STEPS_PER_TASK", 10)
TARGET_TOTAL_STEPS = TARGET_STEPS_PER_TASK * len(TASK_SEQUENCE)
MAX_CURRICULUM_EPISODES = _env_int(
    "MAX_CURRICULUM_EPISODES",
    max(32, TARGET_TOTAL_STEPS * 2),
)
TEMPERATURE = _env_float("TEMPERATURE", 0.0)
MAX_TOKENS = _env_int("MAX_TOKENS", 256)
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 90)
RETRY_WAIT = _env_int("RETRY_WAIT", 5)
ENV_STARTUP_RETRIES = _env_int("ENV_STARTUP_RETRIES", 10)
ENV_STARTUP_WAIT = _env_float("ENV_STARTUP_WAIT", 2.0)

SYSTEM_PROMPT = """You are an expert Sanskrit manuscript interpreter.
Read the passage, question, and candidate options carefully.
Reply with exactly one candidate option copied verbatim.
Do not add explanations or extra text.
""".strip()


def _debug(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _single_line(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clamp_score(value: Optional[float]) -> float:
    try:
        numeric = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return min(max(numeric, 0.0), 1.0)


def _extract_chat_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "".join(chunks).strip()

        return str(content).strip()

    result = payload.get("result")
    if isinstance(result, dict):
        response_text = result.get("response")
        if isinstance(response_text, str):
            return response_text.strip()

    return ""


def _parse_api_error_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "unknown provider error"

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            if isinstance(error_payload, str) and error_payload.strip():
                return error_payload.strip()

            errors = payload.get("errors")
            if isinstance(errors, list):
                messages = []
                for item in errors:
                    if isinstance(item, dict):
                        message = item.get("message")
                        if isinstance(message, str) and message.strip():
                            messages.append(message.strip())
                if messages:
                    return "; ".join(messages)
    except json.JSONDecodeError:
        pass

    return _single_line(text)[:220] or "unknown provider error"


def _select_api_token(base_url: str) -> str:
    lowered = (base_url or "").lower()
    if "cloudflare" in lowered:
        return CLOUDFLARE_API_TOKEN or HF_TOKEN
    if "huggingface" in lowered or "router" in lowered:
        return HF_TOKEN or CLOUDFLARE_API_TOKEN
    return HF_TOKEN or CLOUDFLARE_API_TOKEN


def _llm_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    token = _select_api_token(API_BASE_URL)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _wait_for_env() -> None:
    health_url = f"{ENV_URL}/health"
    last_error = ""

    for attempt in range(ENV_STARTUP_RETRIES):
        try:
            with urllib.request.urlopen(health_url, timeout=10) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return
        except Exception as exc:
            last_error = _single_line(str(exc))
            if attempt < ENV_STARTUP_RETRIES - 1:
                time.sleep(ENV_STARTUP_WAIT)

    raise RuntimeError(f"environment not reachable at {health_url}: {last_error or 'unknown error'}")


def build_user_prompt(obs, rolling_memory: str) -> str:
    lines: List[str] = []

    if getattr(obs, "source_text_iast", ""):
        lines.append(f"Sanskrit (IAST): {obs.source_text_iast}")
    if getattr(obs, "source_text_devanagari", ""):
        lines.append(f"Devanagari: {obs.source_text_devanagari}")
    if getattr(obs, "english_context", ""):
        lines.append(f"Source context: {obs.english_context}")
    if getattr(obs, "domain", ""):
        lines.append(f"Domain: {obs.domain}")
    if getattr(obs, "target_term_iast", None):
        lines.append(f"Term to interpret: {obs.target_term_iast}")
    if getattr(obs, "compound_iast", None):
        label = "Compound to classify" if obs.task_id == "samasa_classification" else "Compound to split"
        lines.append(f"{label}: {obs.compound_iast}")

    verses = getattr(obs, "verses_so_far", None)
    if verses:
        lines.append("")
        lines.append("Verses in this passage:")
        for verse in verses:
            lines.append(f"[{verse['verse_num']}] IAST: {verse['iast']}")
            lines.append(f"English: {verse['english']}")

    if rolling_memory.strip():
        lines.append("")
        lines.append("Previous decisions:")
        lines.append(rolling_memory.strip())

    if getattr(obs, "step_reward", None) and obs.step_reward > 0:
        lines.append("")
        lines.append(f"Previous step reward: {obs.step_reward:.2f}")
    elif getattr(obs, "feedback_message", ""):
        lines.append("")
        lines.append(f"Feedback: {obs.feedback_message}")

    lines.append("")
    lines.append(f"Question: {getattr(obs, 'decision_prompt', '')}")
    lines.append("")
    lines.append("Options:")
    for index, option in enumerate(getattr(obs, "candidate_options", []) or [], start=1):
        lines.append(f"{index}. {option}")
    lines.append("")
    lines.append("Reply with exactly one option.")

    return "\n".join(lines)


def update_rolling_memory(rolling_memory: str, obs, selected_option: str) -> str:
    prompt = getattr(obs, "decision_prompt", "")
    if not prompt:
        return rolling_memory

    summary = f"{prompt.strip().rstrip('?')} -> {selected_option}"
    lines = [line for line in rolling_memory.splitlines() if line.strip()]
    lines.append(summary)
    return "\n".join(lines[-10:])


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not configured")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    request_body = json.dumps(payload).encode("utf-8")

    for attempt in range(4):
        request = urllib.request.Request(
            API_BASE_URL,
            data=request_body,
            headers=_llm_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                text = _extract_chat_text(data)
                if text:
                    return text
                raise RuntimeError("empty model response")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            parsed = _parse_api_error_text(error_body)
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(RETRY_WAIT * (2 ** attempt))
                continue
            raise RuntimeError(f"model request failed ({exc.code}): {parsed}")
        except urllib.error.URLError as exc:
            if attempt < 3:
                time.sleep(RETRY_WAIT * (2 ** attempt))
                continue
            raise RuntimeError(f"model network error: {_single_line(str(exc.reason))}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"model returned non-JSON response: {exc}")

    raise RuntimeError("model retries exhausted")


def match_to_option(raw_answer: str, candidate_options: List[str]) -> str:
    if not candidate_options:
        raise RuntimeError("environment returned no candidate options")

    raw = (raw_answer or "").strip()
    if not raw:
        return candidate_options[0]

    for option in candidate_options:
        if raw == option:
            return option

    numeric_match = re.fullmatch(
        r"(?:option\s*)?[\[(]?([1-9]\d*)[\])\.:\-]?(?:\s+.*)?",
        raw,
        flags=re.IGNORECASE,
    )
    if numeric_match:
        option_index = int(numeric_match.group(1)) - 1
        if 0 <= option_index < len(candidate_options):
            return candidate_options[option_index]

    for option in candidate_options:
        if option.lower().startswith(raw.lower()[:30]):
            return option

    for option in candidate_options:
        if option.lower() in raw.lower():
            return option

    return candidate_options[0]


def choose_action(obs, rolling_memory: str) -> Tuple[str, str, Optional[str]]:
    candidate_options = getattr(obs, "candidate_options", []) or []
    if not candidate_options:
        raise RuntimeError("environment returned no candidate options")

    prompt = build_user_prompt(obs, rolling_memory)
    try:
        raw_answer = call_llm(SYSTEM_PROMPT, prompt)
        return match_to_option(raw_answer, candidate_options), raw_answer, None
    except Exception as exc:
        error = _single_line(str(exc)) or "model request failed"
        _debug(f"model fallback: {error}")
        return candidate_options[0], "", error


def _extract_step_error(obs, model_error: Optional[str]) -> Optional[str]:
    if model_error:
        return model_error

    feedback = _single_line(getattr(obs, "feedback_message", ""))
    lowered = feedback.lower()
    if feedback and ("error" in lowered or "invalid" in lowered or "not found" in lowered):
        return feedback

    return None


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def run_episode(
    env,
    task_id: str,
    seed: int,
    step_offset: int,
    step_limit: int,
) -> Tuple[int, List[float], float, bool]:
    try:
        result = env.reset(task_id=task_id, seed=seed)
        observation = result.observation
    except Exception as exc:
        _debug(f"env.reset failed for task {task_id}: {_single_line(str(exc))}")
        return 0, [], 0.0, False
    rolling_memory = ""
    episode_steps = 0
    episode_rewards: List[float] = []
    task_step_cap = TASK_MAX_STEPS.get(task_id, MAX_STEPS)
    effective_step_limit = max(1, min(task_step_cap, step_limit))

    while not getattr(observation, "done", False) and episode_steps < effective_step_limit:
        current_observation = observation
        global_step = step_offset + episode_steps + 1

        selected_option, raw_answer, model_error = choose_action(current_observation, rolling_memory)
        rolling_memory = update_rolling_memory(rolling_memory, current_observation, selected_option)

        try:
            result = env.step(
                ManuscriptAction(
                    selected_option=selected_option,
                    confidence=0.8,
                    reasoning=raw_answer or model_error or "",
                )
            )
        except Exception as exc:
            step_error = _single_line(str(exc)) or "environment step failed"
            log_step(global_step, selected_option, 0.0, True, step_error)
            episode_steps += 1
            break

        observation = result.observation
        reward = getattr(observation, "step_reward", None)
        if reward is None:
            reward = result.reward
        reward = float(reward if reward is not None else 0.0)
        episode_rewards.append(reward)
        episode_steps += 1

        done = bool(result.done or getattr(observation, "done", False))
        step_error = _extract_step_error(observation, model_error)
        log_step(global_step, selected_option, reward, done, step_error)

        if done:
            break

    episode_score = _clamp_score(getattr(observation, "cumulative_score", 0.0))
    if episode_score == 0.0 and getattr(result, "reward", None) is not None:
        episode_score = _clamp_score(result.reward)

    episode_success = bool(getattr(observation, "done", False)) and episode_score > 0.0
    return episode_steps, episode_rewards, episode_score, episode_success


def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    display_task_name = "all" if RUN_CURRICULUM else TASK_NAME
    log_start(task=display_task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        _wait_for_env()

        with SanskritEnv(base_url=ENV_URL).sync() as env:
            if RUN_CURRICULUM:
                task_step_counts = {task_id: 0 for task_id in TASK_SEQUENCE}
                task_weighted_scores = {task_id: 0.0 for task_id in TASK_SEQUENCE}
                task_score_weights = {task_id: 0 for task_id in TASK_SEQUENCE}
                task_pointer = 0

                for episode_index in range(MAX_CURRICULUM_EPISODES):
                    if all(
                        task_step_counts[task_id] >= TARGET_STEPS_PER_TASK
                        for task_id in TASK_SEQUENCE
                    ):
                        break

                    task_id = None
                    for _ in range(len(TASK_SEQUENCE)):
                        candidate_task = TASK_SEQUENCE[task_pointer % len(TASK_SEQUENCE)]
                        task_pointer += 1
                        if task_step_counts[candidate_task] < TARGET_STEPS_PER_TASK:
                            task_id = candidate_task
                            break

                    if task_id is None:
                        break

                    remaining_steps_for_task = TARGET_STEPS_PER_TASK - task_step_counts[task_id]
                    episode_steps, episode_rewards, episode_score, _ = run_episode(
                        env=env,
                        task_id=task_id,
                        seed=RANDOM_SEED + episode_index,
                        step_offset=steps_taken,
                        step_limit=remaining_steps_for_task,
                    )

                    if episode_steps <= 0:
                        break

                    steps_taken += episode_steps
                    rewards.extend(episode_rewards)
                    task_step_counts[task_id] += episode_steps
                    task_weighted_scores[task_id] += episode_score * episode_steps
                    task_score_weights[task_id] += episode_steps

                task_scores = []
                for task_id in TASK_SEQUENCE:
                    if task_score_weights[task_id] > 0:
                        task_scores.append(task_weighted_scores[task_id] / task_score_weights[task_id])

                if task_scores:
                    score = _clamp_score(sum(task_scores) / len(task_scores))
                    success = all(
                        task_step_counts[task_id] >= TARGET_STEPS_PER_TASK
                        for task_id in TASK_SEQUENCE
                    ) and score > 0.0
            else:
                episode_steps, episode_rewards, episode_score, episode_success = run_episode(
                    env=env,
                    task_id=TASK_NAME,
                    seed=RANDOM_SEED,
                    step_offset=0,
                    step_limit=MAX_STEPS,
                )
                steps_taken = episode_steps
                rewards.extend(episode_rewards)
                score = episode_score
                success = episode_success

    except Exception as exc:
        _debug(f"inference error: {_single_line(str(exc))}")
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
