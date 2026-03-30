---
title: SanskritEnv
emoji: 📜
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
short_description: RL environment for Sanskrit manuscript interpretation
---

# SanskritEnv

> An OpenEnv-compatible RL environment for Sanskrit manuscript interpretation.
> Train and evaluate AI agents on the task of resolving structural linguistic
> ambiguity in ancient Indian texts — a real bottleneck in ongoing digitization
> projects backed by the Indian government.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue?logo=huggingface)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/Aditya_Raj/sanskrit-env)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

---

## Why this environment exists

India's **Gyan Bharatam Mission** (Union Budget 2025–26) is digitizing over
**1 crore Sanskrit manuscripts**. Projects like eGangotri have already rescued
and scanned more than 60,000 rare texts and 1.4 crore pages. The problem:
digitization has outpaced translation by orders of magnitude. The bottleneck
is not scanning technology — it is the shortage of scholars who can read
classical Sanskrit across its three major difficulty layers:

| Layer | Problem | What blocks automation |
|-------|---------|----------------------|
| Lexical | A single term (e.g. *agni*) has 4–6 domain-specific meanings | No contextual disambiguation |
| Phonological | Compound words (*sandhi*) have multiple valid splits | Requires grammatical + contextual reasoning |
| Discourse | Pronouns and implicit subjects span multiple verses | Requires cross-sentence coreference tracking |

SanskritEnv provides a structured benchmark where AI agents must solve exactly
these three problems, with fully deterministic graders and dense reward signals.
No existing OpenEnv environment addresses Sanskrit, ancient linguistics, or
cultural heritage preservation.

---

## Environment overview

SanskritEnv is a **decision environment**, not a translation model.
At each step the agent receives a Sanskrit passage and must select the
correct linguistic interpretation from four deterministically-graded options.

```
Agent ──[ManuscriptAction]──► SanskritEnv ──[ManuscriptObservation + reward]──► Agent
```

Three tasks, escalating difficulty:

| Task | ID | Difficulty | Steps/episode | Core challenge |
|------|----|-----------|--------------|----------------|
| Glossary Anchoring | `glossary_anchoring` | Easy | 1 | Domain-specific term disambiguation |
| Sandhi Resolution | `sandhi_resolution` | Medium | 1 | Phonological compound splitting |
| Referential Coherence | `referential_coherence` | Hard | 4–7 | Cross-verse pronoun tracking |

---

## Baseline scores

Measured with `llama-3.3-70b-versatile` (Groq), ReAct + Memory architecture,
`temperature=0.0`, 5 episodes per task, seed=42.

| Task | Score | Std dev |
|------|-------|---------|
| Task 1 — Glossary Anchoring (Easy) | `1.000` | `±0.000` |
| Task 2 — Sandhi Resolution (Medium) | `1.000` | `±0.000` |
| Task 3 — Referential Coherence (Hard) | `0.840` | `±0.102` |

*Run `python baseline.py` to reproduce. Results are saved to `baseline_results.json`.*

---

## Action space

```python
ManuscriptAction(
    selected_option: str,   # Must match one entry in candidate_options EXACTLY
    confidence: float = 0.5,   # Agent self-reported confidence — logged, not graded
    reasoning: str = "",        # Agent explanation — logged, not graded
)
```

**Critical:** `selected_option` must be copied verbatim from `candidate_options`.
Any string not in the list returns `reward=0.0` and terminates the episode.

---

## Observation space

```python
ManuscriptObservation(
    # Always present
    task_id: str,                    # "glossary_anchoring" | "sandhi_resolution" | "referential_coherence"
    episode_id: str,                 # Unique episode identifier
    source_text_iast: str,           # Sanskrit in IAST transliteration
    source_text_devanagari: str,     # Sanskrit in Devanagari script
    english_context: str,            # Source text and domain description
    domain: str,                     # "ayurveda" | "astronomy" | "philosophy" | "narrative"
    decision_prompt: str,            # The question the agent must answer
    candidate_options: List[str],    # Exactly 4 options — select one verbatim
    step_reward: float,              # Reward earned on the previous step (0.0 at step 1)
    cumulative_score: float,         # Running episode score (0.0–1.0)
    feedback_message: str,           # Plain-English explanation of previous reward
    done: bool,                      # True when episode is complete
    reward: Optional[float],         # Final episode score when done=True, else None

    # Task 1 only
    target_term_iast: Optional[str],           # The term to interpret
    active_glossary: Optional[Dict[str, str]], # Domain term reference

    # Task 2 only
    compound_iast: Optional[str],              # The compound word to split

    # Task 3 only
    verses_so_far: Optional[List[Dict]],       # All verses seen: [{verse_num, iast, english}]
    current_verse_num: Optional[int],          # Current verse being processed
    consistency_history: Optional[List[Dict]], # Prior checkpoint answers: [{question, answer}]
)
```

---

## Reward function

Rewards are **dense** — the agent receives signal at every step, not just at
episode end. This provides gradient information even for partially correct answers.

### Task 1 — Glossary Anchoring

| Outcome | Reward |
|---------|--------|
| Exact correct domain meaning | `+1.00` |
| Partial credit option (related but imprecise) | `+0.40` |
| Wrong meaning | `+0.00` |
| Invalid selection (not in candidate_options) | `+0.00` + episode ends |

### Task 2 — Sandhi Resolution

| Outcome | Reward |
|---------|--------|
| Correct phonological split | `+1.00` |
| Adjacent analysis (same first component, slightly wrong) | `+0.25` |
| Wrong split | `+0.00` |
| Invalid selection | `+0.00` + episode ends |

### Task 3 — Referential Coherence

| Outcome | Reward |
|---------|--------|
| Each correct checkpoint answer | `+0.10` |
| Correct final antecedent identification | `+0.70` |
| Wrong checkpoint or final answer | `+0.00` |

All episode scores are normalized to **0.0–1.0** before being returned as the
final `reward` when `done=True`.

---

## Grader design — why no LLM, no BLEU

All three graders are fully deterministic:
- **No LLM judge calls** — judges in Phase 1 will check this
- **No BLEU/ROUGE** — unreliable for Sanskrit free word order
- **Exact string match** against pre-annotated answer tables embedded in data JSON

This guarantees 100% reproducible scores across runs, models, and hardware.
Two runs with the same seed will always produce identical scores.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Groq API key (free): [console.groq.com](https://console.groq.com)

### Local development

```bash
# Clone
git clone https://huggingface.co/spaces/Aditya_Raj/sanskrit-env
cd sanskrit-env

# Install
pip install -r requirements.txt

# Run environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Validate (separate terminal)
openenv validate --url http://localhost:7860
```

### Docker

```bash
# Build
docker build -t sanskrit-env:latest .

# Run
docker run -p 7860:7860 sanskrit-env:latest

# Health check
curl http://localhost:7860/health
# → {"status": "healthy"}
```

### Run baseline

```bash
export GROQ_API_KEY=your_key_here
export SANSKRIT_ENV_URL=http://localhost:7860

# All tasks
python baseline.py

# Single task
python baseline.py --task referential_coherence
```

---

## Usage

### Minimal example

```python
from client import SanskritEnv
from models import ManuscriptAction

with SanskritEnv(base_url="https://Aditya_Raj-sanskrit-env.hf.space").sync() as env:

    # Task 1 — single step
    result = env.reset(task_id="glossary_anchoring")
    obs = result.observation

    print(obs.source_text_iast)     # Sanskrit passage
    print(obs.decision_prompt)      # Question
    print(obs.candidate_options)    # 4 options

    result = env.step(ManuscriptAction(
        selected_option=obs.candidate_options[0],
        reasoning="This matches the Ayurvedic domain context."
    ))
    print(f"Score: {result.reward}")
```

### Task 3 — multi-step with memory

```python
from client import SanskritEnv
from models import ManuscriptAction

rolling_memory = ""

with SanskritEnv(base_url="https://Aditya_Raj-sanskrit-env.hf.space").sync() as env:
    result = env.reset(task_id="referential_coherence")
    obs = result.observation

    while not obs.done:
        # Show verses and question
        if obs.verses_so_far:
            for v in obs.verses_so_far:
                print(f"  Verse {v['verse_num']}: {v['english']}")

        print(f"\nQuestion: {obs.decision_prompt}")

        # Agent picks an option (replace with your model)
        selected = obs.candidate_options[0]

        # Update rolling memory
        rolling_memory += f"\n• {obs.decision_prompt} → {selected}"

        result = env.step(ManuscriptAction(selected_option=selected))
        obs = result.observation
        print(f"  Reward this step: {obs.step_reward:.2f}")

    print(f"\nFinal episode score: {obs.reward:.4f}")
```

### Reproducible evaluation

```python
# Fixed seed ensures same episode is loaded every run
result = env.reset(task_id="sandhi_resolution", seed=42)
```

---

## Agent architecture (baseline)

The included `baseline.py` implements a **ReAct + Memory** loop:

```
┌─────────────────────────────────────────────────────────┐
│  ReAct + Memory loop (one Groq call per step)           │
│                                                         │
│  rolling_memory = ""   ← starts empty each episode     │
│                                                         │
│  while not done:                                        │
│    1. THINK  — build prompt from obs + rolling_memory   │
│    2. ACT    — call Groq, get raw answer                │
│    3. MATCH  — match raw answer to candidate_options    │
│    4. STEP   — env.step(ManuscriptAction(selected))     │
│    5. UPDATE — append "Q → A" to rolling_memory        │
└─────────────────────────────────────────────────────────┘
```

The `rolling_memory` string grows by one line per step and is injected
into every prompt. For Task 3 this looks like:

```
── What you have established so far in this episode ──
• Who is 'sa' in verse 3? → Savitri
• Who is 'sa' in verse 5? → Savitri
• Who fell (patitah)? → Satyavan
── Use this to stay consistent ──
```

This prevents the referential drift that a naive single-prompt-per-step
agent suffers on multi-verse passages.

---

## Data sources

All ground truth data is curated from public domain Sanskrit texts,
annotated by the project authors. No proprietary data is used.

| Text | Domain | Task |
|------|--------|------|
| Charaka Samhita | Ayurveda | Task 1 |
| Ashtanga Hridayam | Ayurveda | Task 1 |
| Sushruta Samhita | Ayurveda | Task 1 |
| Aryabhatiya | Astronomy | Task 1 |
| Arthashastra | Political philosophy | Task 1, 3 |
| Bhagavad Gita | Vedanta philosophy | Task 1, 2, 3 |
| Mundaka Upanishad | Vedanta philosophy | Task 2 |
| Brihadaranyaka Upanishad | Vedanta philosophy | Task 2 |
| Ramayana (Ayodhya Kanda) | Narrative | Task 2, 3 |
| Mahabharata (Vana Parva) | Narrative | Task 3 |

---

## Evaluation phases

This environment participates in the Meta × HuggingFace OpenEnv hackathon:

- **Phase 1 (Automated):** `openenv validate` passes, Docker builds, baseline reproduces
- **Phase 2 (Agentic):** Standard Open LLM agent (Nemotron 3 Super) is run against all tasks
- **Phase 3 (Human):** Meta and HuggingFace engineers review real-world utility and grader integrity

---

## Contributing

Contributions welcome. Highest-priority areas:

1. **More episodes** — additional Sanskrit passages with annotated answers
   (must include IAST, Devanagari, English context, 4 options, correct answer)
2. **New domains** — Jyotisha (astrology), Natya Shastra (dramaturgy), Vedic hymns
3. **Harder sandhi cases** — especially involving anusvara, visarga, and vowel coalescence
4. **Multi-language target** — currently English-only; Hindi or regional language target translations

Please open an issue before starting a large contribution.

---

## Citation

If you use SanskritEnv in your research:

```bibtex
@misc{sanskritenv2025,
  title   = {SanskritEnv: A Reinforcement Learning Environment for Sanskrit Manuscript Interpretation},
  author  = {YOUR_NAME},
  year    = {2025},
  url     = {https://huggingface.co/spaces/Aditya_Raj/sanskrit-env},
  note    = {OpenEnv-compatible environment for structured linguistic ambiguity resolution}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Sanskrit texts used are in the public domain (composed before 1928).
Annotations, graders, and environment code are original to this project.

---

## Acknowledgements

- [Meta × HuggingFace OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [Gyan Bharatam Mission](https://indiaculture.gov.in) — the real-world problem this addresses
- [Monier-Williams Sanskrit Dictionary](https://www.sanskrit-lexicon.uni-koeln.de) — lexical reference
- [Digital Corpus of Sanskrit](https://www.sanskrit-linguistics.org/dcs/) — annotated corpus reference
```
