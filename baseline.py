"""
baseline.py — SanskritEnv Baseline Inference Script (HF Router API + ReAct + Memory)

Architecture: ReAct + Memory loop
  Think   -> agent reasons using full verse context + rolling_memory of prior decisions
  Act     -> agent selects one candidate_option verbatim
  Observe -> environment returns reward + feedback_message
  Update  -> agent appends a one-line referent summary to rolling_memory

LLM backend: Hugging Face Router (token auth via HF_TOKEN)
Token: https://huggingface.co/settings/tokens

Usage:
    # Option 1: set env vars directly
    set HF_TOKEN=your_hf_token_here
    set SANSKRIT_ENV_URL=http://localhost:7860

    # Option 2: place values in .env (loaded automatically)
    # HF_TOKEN=your_hf_token_here
    # SANSKRIT_ENV_URL=http://localhost:7860

    python baseline.py
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import List

from dotenv import load_dotenv

from client import SanskritEnv
from models import ManuscriptAction

# ── Configuration ────────────────────────────────────────────────────────────

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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
ENV_URL = os.environ.get("SANSKRIT_ENV_URL", "http://localhost:7860")
DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_ROUTER_URL = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")

HF_MODEL_CANDIDATES = [
    model.strip()
    for model in os.environ.get(
        "HF_MODEL_CANDIDATES",
        "Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-7B-Instruct,google/gemma-2-9b-it,mistralai/Mistral-7B-Instruct-v0.3,HuggingFaceH4/zephyr-7b-beta",
    ).split(",")
    if model.strip()
]
AUTO_MODEL_FALLBACK = _env_bool("HF_AUTO_MODEL_FALLBACK", True)

TEMPERATURE = _env_float("HF_TEMPERATURE", 0.0)
MAX_TOKENS = _env_int("HF_MAX_TOKENS", 512)
EPISODES_PER_TASK = _env_int("EPISODES_PER_TASK", 15)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
RETRY_WAIT = _env_int("RETRY_WAIT", 5)
REQUEST_TIMEOUT = _env_int("HF_REQUEST_TIMEOUT", 90)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Sanskrit manuscript interpreter with deep knowledge of:
- Classical Sanskrit grammar, phonology, and sandhi rules
- Paninian grammar: samasa (compound) classification — Tatpurusha, Karmadharaya, Dvigu, Dvandva, Bahuvrihi, Avyayibhava
- Ayurvedic texts: Charaka Samhita, Sushruta Samhita, Ashtanga Hridayam
- Astronomical texts: Aryabhatiya, Brahmasphutasiddhanta
- Philosophical texts: Bhagavad Gita, Upanishads, Vivekachudamani
- Narrative texts: Ramayana, Mahabharata, Arthashastra

For samasa classification, apply these rules:
- Tatpurusha: second member is the semantic head, first member qualifies it via a case relation
- Karmadharaya: adjective + noun referring to the SAME entity (subtype of tatpurusha)
- Dvigu: numeral + noun forming a collective (subtype of tatpurusha)
- Dvandva: both members are coordinate and equal — "A and B" — dual/plural number reflects sum
- Bahuvrihi: compound is adjectival, describes an EXTERNAL referent — neither member is the head
- Avyayibhava: whole compound is an indeclinable adverb, first member is usually a prefix/indeclinable

Your task each step:
1. THINK: reason carefully about the Sanskrit passage and question
2. SELECT: choose EXACTLY ONE option from the list provided
3. OUTPUT: respond with ONLY the exact text of your chosen option — nothing else

Rules:
- Your entire response must be one of the provided options, copied character-for-character
- Do not add explanation, punctuation, or any other text
- If unsure, pick the option that best fits the domain and grammatical context"""

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_user_prompt(obs, rolling_memory: str) -> str:
    """
    Build the full prompt for one ReAct step.

    Injects rolling_memory so the agent has access to all prior decisions
    established in this episode — critical for Task 4 coherence tracking.
    """
    lines = []

    # Source text
    if obs.source_text_iast:
        lines.append(f"Sanskrit (IAST): {obs.source_text_iast}")
    if obs.source_text_devanagari:
        lines.append(f"Devanagari:      {obs.source_text_devanagari}")
    if obs.english_context:
        lines.append(f"Source context:  {obs.english_context}")
    if obs.domain:
        lines.append(f"Domain:          {obs.domain}")

    # Task-specific fields
    if obs.target_term_iast:
        lines.append(f"Term to interpret: {obs.target_term_iast}")
    if obs.compound_iast:
        label = "Compound to classify" if obs.task_id == "samasa_classification" else "Compound to split"
        lines.append(f"{label}: {obs.compound_iast}")

    # Task 4: full verse history
    if obs.verses_so_far:
        lines.append("")
        lines.append("Verses in this passage:")
        for v in obs.verses_so_far:
            lines.append(f"  [{v['verse_num']}] IAST:    {v['iast']}")
            lines.append(f"       English: {v['english']}")

    # ReAct Memory — inject everything established so far in this episode
    if rolling_memory.strip():
        lines.append("")
        lines.append("── What you have established so far in this episode ──")
        lines.append(rolling_memory.strip())
        lines.append("── Use this to stay consistent ──")

    # Reward signal from last step (helps agent self-correct)
    if obs.step_reward and obs.step_reward > 0:
        lines.append("")
        lines.append(f"Your last answer was CORRECT (reward: {obs.step_reward:.2f}).")
    elif obs.step_reward == 0.0 and obs.feedback_message:
        lines.append("")
        lines.append(f"Feedback: {obs.feedback_message}")

    # The decision
    lines.append("")
    lines.append(f"Question: {obs.decision_prompt}")
    lines.append("")
    lines.append("Options (choose one exactly as written):")
    for i, opt in enumerate(obs.candidate_options):
        lines.append(f"  {i + 1}. {opt}")
    lines.append("")
    lines.append("Your answer (exact option text only):")

    return "\n".join(lines)


def update_rolling_memory(rolling_memory: str, obs, selected_option: str) -> str:
    """
    Append a one-line summary of the just-completed decision to rolling_memory.

    This is the 'Update' phase of the ReAct + Memory loop.
    For Task 4 specifically, this records which character/entity each
    pronoun or implicit subject refers to, so future steps can stay consistent.
    """
    if not obs.decision_prompt:
        return rolling_memory

    # Build a concise one-liner
    summary = f"• {obs.decision_prompt.strip().rstrip('?')} → {selected_option}"

    # Cap at 10 lines to prevent prompt bloat
    lines = [l for l in rolling_memory.strip().split("\n") if l.strip()]
    lines.append(summary)
    if len(lines) > 10:
        lines = lines[-10:]

    return "\n".join(lines)


# ── HF Router API call with retry ────────────────────────────────────────────

def _extract_router_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    # Some models return content as an array of chunks.
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks).strip()

    return str(content).strip()


def _parse_router_error_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "unknown provider error"

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            if isinstance(err, str) and err.strip():
                return err.strip()
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    title_start = lowered.find("<title>")
    if title_start != -1:
        title_end = lowered.find("</title>", title_start)
        if title_end != -1:
            title = text[title_start + len("<title>"):title_end].strip()
            if title:
                return title

    return " ".join(text.split())[:220]


def _is_credits_exhausted(message: str) -> bool:
    msg = (message or "").lower()
    return "depleted your monthly included credits" in msg or "purchase pre-paid credits" in msg


def _probe_model_access(model: str) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with OK only."},
            {"role": "user", "content": "OK"},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }

    request_body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        HF_ROUTER_URL,
        data=request_body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=min(REQUEST_TIMEOUT, 30)) as response:
            if 200 <= response.status < 300:
                return True, "ok"
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        parsed = _parse_router_error_text(error_body)
        return False, f"{exc.code}: {parsed}"
    except urllib.error.URLError as exc:
        return False, f"network: {exc.reason}"
    except Exception as exc:
        return False, f"error: {exc}"


def select_model_for_run(requested_model: str) -> str:
    ordered: List[str] = [requested_model]
    ordered.extend(m for m in HF_MODEL_CANDIDATES if m != requested_model)

    print("Checking HF model availability for current token/provider...")
    unavailable = []

    for model in ordered:
        ok, detail = _probe_model_access(model)
        if ok:
            if model == requested_model:
                print(f"  Using requested model: {model}")
            else:
                print(f"  Requested model unavailable; auto-fallback to: {model}")
            return model

        unavailable.append((model, detail))
        print(f"  Unavailable: {model} ({detail})")

        if _is_credits_exhausted(detail):
            raise RuntimeError(
                f"HF Router credits exhausted for this token. {detail}"
            )

    summary = "; ".join(f"{model} -> {reason}" for model, reason in unavailable[:4])
    raise RuntimeError(
        "No available HF chat model found for this token/provider setup. "
        f"Tried: {summary}"
    )


def call_llm(model: str, system: str, user: str) -> str:
    """
    Call Hugging Face Router chat completions endpoint with exponential backoff.
    Auth uses HF_TOKEN, not any OpenAI/Groq client SDK.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    request_body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    for attempt in range(4):
        req = urllib.request.Request(
            HF_ROUTER_URL,
            data=request_body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                return _extract_router_text(data)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            if exc.code in (429, 500, 502, 503, 504):
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"    [router {exc.code}] waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
                continue
            parsed = _parse_router_error_text(error_body)
            raise RuntimeError(f"Router request failed ({exc.code}): {parsed}")
        except urllib.error.URLError as exc:
            wait = RETRY_WAIT * (2 ** attempt)
            print(f"    [network] {exc.reason}; waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Router returned non-JSON response: {exc}")

    return ""


# ── Option matching ───────────────────────────────────────────────────────────

def match_to_option(raw_answer: str, candidate_options: list) -> str:
    """
    Match the LLM's raw output to the closest candidate_option.

    Priority:
    1. Exact match
    2. Candidate starts with the raw answer (model truncated)
    3. Raw answer starts with the candidate (model added padding)
    4. Random fallback (prevents crash, penalised by grader)
    """
    raw = raw_answer.strip()

    # 1. Exact
    for opt in candidate_options:
        if raw == opt:
            return opt

    # 2. Prefix: model gave first N chars of option
    for opt in candidate_options:
        if opt.lower().startswith(raw.lower()[:30]):
            return opt

    # 3. Contains: raw answer contains the option
    for opt in candidate_options:
        if opt.lower() in raw.lower():
            return opt

    # 4. Random fallback
    print(f"    [warn] could not match '{raw[:60]}' to any option — random fallback")
    return random.choice(candidate_options)


# ── Episode runner (ReAct + Memory loop) ─────────────────────────────────────

def run_episode(env, model: str, task_id: str, seed: int, verbose: bool = True) -> float:
    """
    Run one complete episode using the ReAct + Memory architecture.

    Loop:
        while not done:
            user_prompt = build_user_prompt(obs, rolling_memory)
            raw_answer  = call_llm(model, system, user_prompt)
            selected    = match_to_option(raw_answer, obs.candidate_options)
            result      = env.step(ManuscriptAction(selected_option=selected, reasoning=raw_answer))
            rolling_memory = update_rolling_memory(rolling_memory, obs, selected)
            obs = result.observation

    Returns the final episode score (0.0–1.0).
    """
    result = env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    rolling_memory = ""   # starts empty every episode
    step = 0

    while not obs.done:
        step += 1
        user_prompt = build_user_prompt(obs, rolling_memory)
        raw_answer  = call_llm(model, SYSTEM_PROMPT, user_prompt)
        selected    = match_to_option(raw_answer, obs.candidate_options)

        if verbose:
            print(f"    Step {step}: selected → '{selected[:60]}'")

        # Update memory BEFORE stepping (so the summary is of current decision)
        rolling_memory = update_rolling_memory(rolling_memory, obs, selected)

        result = env.step(ManuscriptAction(
            selected_option=selected,
            confidence=0.8,
            reasoning=raw_answer,
        ))
        obs = result.observation

        if obs.step_reward is not None and verbose:
            print(f"            reward: {obs.step_reward:.2f} | cumulative: {obs.cumulative_score:.2f}")

    final_score = obs.reward if obs.reward is not None else 0.0
    if verbose:
        print(f"    Episode done — final score: {final_score:.4f}")

    return final_score


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, label: str, model: str) -> dict:
    # Task 1 — glossary_anchoring  (Easy)
    # Task 2 — sandhi_resolution   (Medium)
    # Task 3 — samasa_classification (Medium)
    # Task 4 — referential_coherence (Hard)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Model: {model} | Episodes: {EPISODES_PER_TASK} | Seed base: {RANDOM_SEED}")
    print(f"{'='*65}")

    scores = []
    with SanskritEnv(base_url=ENV_URL).sync() as env:
        for i in range(EPISODES_PER_TASK):
            seed = RANDOM_SEED + i
            print(f"\n  Episode {i + 1}/{EPISODES_PER_TASK} (seed={seed})")
            try:
                score = run_episode(env, model, task_id, seed, verbose=True)
                scores.append(score)
            except Exception as exc:
                message = str(exc)
                print(f"  ERROR: {message}")
                scores.append(0.0)
                if _is_credits_exhausted(message):
                    remaining = EPISODES_PER_TASK - len(scores)
                    if remaining > 0:
                        scores.extend([0.0] * remaining)
                    print("  Stopping task early: HF Router credits are exhausted.")
                    break

    mean   = sum(scores) / len(scores)
    stddev = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    print(f"\n  Scores:  {[round(s, 3) for s in scores]}")
    print(f"  Mean:    {mean:.4f}")
    print(f"  Std dev: {stddev:.4f}")

    return {
        "task_id":  task_id,
        "label":    label,
        "model":    model,
        "episodes": EPISODES_PER_TASK,
        "seed":     RANDOM_SEED,
        "scores":   scores,
        "mean":     round(mean, 4),
        "stddev":   round(stddev, 4),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SanskritEnv baseline inference (HF Router API + ReAct + Memory)")
    parser.add_argument(
        "--task",
        choices=["glossary_anchoring", "sandhi_resolution", "samasa_classification", "referential_coherence", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model ID to use (default from HF_MODEL or Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--no-auto-fallback",
        action="store_true",
        help="Disable automatic fallback to an available HF model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_TASK,
        help="Episodes per task (default from EPISODES_PER_TASK env or 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed base (default from RANDOM_SEED env or 42)",
    )
    args = parser.parse_args()

    EPISODES_PER_TASK = args.episodes
    RANDOM_SEED = args.seed

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("  Get a free HF token at: https://huggingface.co/settings/tokens")
        print("  Or set HUGGINGFACEHUB_API_TOKEN in .env")
        sys.exit(1)

    effective_model = args.model
    auto_fallback_enabled = AUTO_MODEL_FALLBACK and not args.no_auto_fallback
    if auto_fallback_enabled:
        try:
            effective_model = select_model_for_run(args.model)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            print("Hint: try a supported model explicitly, e.g. --model Qwen/Qwen2.5-72B-Instruct")
            sys.exit(1)
    elif args.no_auto_fallback:
        print("Model fallback disabled via --no-auto-fallback")
    else:
        print("Model fallback disabled via HF_AUTO_MODEL_FALLBACK=0")

    print(f"\nSanskritEnv Baseline — HF Router API + ReAct + Memory")
    print(f"Environment: {ENV_URL}")
    print(f"Model:       {effective_model}")
    print(f"Router URL:  {HF_ROUTER_URL}")
    print(f"Architecture: ReAct + rolling_memory (Think->Act->Observe->Update)")

    # Canonical task ordering:
    #   Task 1 — glossary_anchoring       (Easy)
    #   Task 2 — sandhi_resolution        (Medium)
    #   Task 3 — samasa_classification    (Medium)
    #   Task 4 — referential_coherence    (Hard)
    tasks_to_run = {
        "glossary_anchoring":    "Task 1 — Glossary Anchoring (Easy)",
        "sandhi_resolution":     "Task 2 — Sandhi Resolution (Medium)",
        "samasa_classification": "Task 3 — Samasa Classification (Medium)",
        "referential_coherence": "Task 4 — Referential Coherence (Hard)",
    }

    if args.task != "all":
        tasks_to_run = {args.task: tasks_to_run[args.task]}

    results = []
    for task_id, label in tasks_to_run.items():
        results.append(run_task(task_id, label, effective_model))

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL BASELINE RESULTS")
    print(f"{'='*65}")
    for r in results:
        bar = "█" * int(r["mean"] * 20)
        print(f"  {r['label']}")
        print(f"    {bar:<20} {r['mean']:.4f} ± {r['stddev']:.4f}")
    print(f"{'='*65}\n")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    print("Copy the mean scores into README.md baseline table.")
