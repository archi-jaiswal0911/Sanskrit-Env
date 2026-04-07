---
title: SanskritEnv
emoji: 📜
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
short_description: RL environment for Sanskrit manuscript interpretation
huggingFace_url: https://huggingface.co/spaces/Adityahars/Sanskrit-env
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

## Real-World Impact

India possesses an estimated **1 crore Sanskrit manuscripts** written in over 80 scripts and 60 languages — the largest manuscript collection of any civilisation on Earth. The **Union Budget 2025-26** allocated ₹60 crore to digitize over **1 crore of these manuscripts** under the **Gyan Bharatam Mission**. As of 2025, metadata for 52 lakh manuscripts has been recorded — but only 1.3 lakh have been uploaded online. Digitization is accelerating, translation is not.

The reason is a collapse in human expertise. Trained Sanskrit scholars capable of reading classical manuscripts are retiring faster than new scholars can replace them. The Government's own National Mission for Manuscripts states directly: *"Scholars who can study and use manuscripts are fast disappearing and a new generation of scholars is not able to rise to the challenge."* A nationwide survey launched in 2026 confirmed the crisis is active and growing.

The three exact linguistic problems that block automated translation of these manuscripts are:
- A single Sanskrit term can carry 4-6 domain-specific meanings with no contextual signal (lexical ambiguity).
- Compound words have multiple valid phonological splits with different meanings (sandhi and samasa ambiguity).
- Pronouns and implicit subjects span multiple verses with no explicit antecedent markers (referential ambiguity).

SanskritEnv is the first RL environment built to train agents on exactly these three problems — using real passages from Ayurvedic, astronomical, philosophical, and narrative manuscripts that are currently sitting in India's national repositories.

---

## How this Environment Solves the Problem

Projects like eGangotri have already rescued
and scanned more than 60,000 rare texts and 1.4 crore pages. The problem:
digitization has outpaced translation by orders of magnitude. The bottleneck
is not scanning technology — it is the shortage of scholars who can read
classical Sanskrit across its four major difficulty layers:

| Layer | Problem | What blocks automation |
|-------|---------|----------------------|
| Lexical | A single term (e.g. *agni*) has 4–6 domain-specific meanings | No contextual disambiguation |
| Phonological | Compound words (*sandhi*) have multiple valid splits | Requires grammatical + contextual reasoning |
| Morphological | Compound words (samāsa) must be classified before they can be parsed | Requires grammatical meta-knowledge |
| Discourse | Pronouns and implicit subjects span multiple verses | Requires cross-sentence coreference tracking |

SanskritEnv provides a structured benchmark where AI agents must solve exactly
these four problems, with fully deterministic graders and dense reward signals.
No existing OpenEnv environment addresses Sanskrit, ancient linguistics, or
cultural heritage preservation.

India's National Mission for Manuscripts has catalogued over 5.2 million manuscripts across 51 cataloguing centres; fewer than 1% have been translated into any modern language. The ratio of trained Sanskrit scholars capable of reading classical manuscripts to the volume of digitized texts is estimated at 1:10,000 and widening every year as digitization accelerates. The four linguistic layers modeled in SanskritEnv — lexical, morphological, phonological, and discourse — are the same four layers cited by Murugesh et al. (2019) "A Survey of Sanskrit NLP" as the primary obstacles to automated translation pipeline construction. SanskritEnv is the first OpenEnv environment targeting ancient-language manuscript interpretation, filling a gap that is both culturally significant and computationally underexplored.

---

## Environment overview

SanskritEnv is a **decision environment**, not a translation model.
At each step the agent receives a Sanskrit passage and must select the
correct linguistic interpretation from four deterministically-graded options.

```
Agent ──[ManuscriptAction]──► SanskritEnv ──[ManuscriptObservation + reward]──► Agent
```

<img width="600" alt="Meta_Mesh" src="https://github.com/user-attachments/assets/90b5dc8b-0b55-46c3-9ee0-5afc856b042a" />



Four tasks, escalating difficulty:

| Task | ID | Difficulty | Steps/episode | Core challenge |
|------|----|-----------|--------------|----------------|
| Glossary Anchoring | `glossary_anchoring` | Easy | 1 | Domain-specific term disambiguation |
| Sandhi Resolution | `sandhi_resolution` | Medium | 1 | Phonological compound splitting |
| Samāsa Classification | `samasa_classification` | Medium | 1 | Grammatical compound type identification |
| Referential Coherence | `referential_coherence` | Hard | 4–7 | Cross-verse pronoun tracking |

---

## Data sources

All ground truth data is curated from public domain Sanskrit texts,
annotated by the project authors. No proprietary data is used.

| Text | Domain | Task | Links |
|------|--------|------|-------|
| Sushruta Samhita | Ayurveda | Task 1 | http://niimh.res.in/ebooks/esushruta/?mod=read |
| Bhagavad Gita | Vedanta philosophy | Task 1, 2, 4 | https://sanskritdocuments.org/sanskrit/bhagavadgita/ |
| Charaka Samhita | Ayurveda | Task 1, 3 | https://niimh.nic.in/ebooks/ecaraka/index.php | 
| Ashtanga Hridayam | Ayurveda | Task 1, 3 | https://archive.org/details/Ashtanga.Hridaya.of.Vagbhata/page/n463/mode/2up |
| Aryabhatiya | Astronomy | Task 1, 3 | https://archive.org/details/Aryabhatiya1976/Aryabhatiya%20v1%201976/ |
| Arthashastra | Political philosophy | Task 1, 3, 4 | https://archive.org/details/in.ernet.dli.2015.485591/page/131/mode/2up |
| Mundaka Upanishad | Vedanta philosophy | Task 2 | https://sanskritdocuments.org/doc_upanishhat/mundaka.html |
| Brihadaranyaka Upanishad | Vedanta philosophy | Task 2 | https://sanskritdocuments.org/doc_upanishhat/brinew-proofed.html |
| Ramayana (Ayodhya Kanda) | Narrative | Task 2, 3, 4 | https://archive.org/details/ValmikiRamayana-AyodhyaKandaWithGovindarajaCommentary/page/%E0%A5%A7%E0%A5%AC%E0%A5%AC/mode/2up |
| Vishnu Sahasranama | Philosophy | Task 3 | https://www.swami-krishnananda.org/vishnu/vishnu_4.html |
| Meghaduta (Kalidasa) | Narrative | Task 3 |https://sanskritdocuments.org/doc_z_misc_major_works/meghanew.html |
| Mahabharata (Vana Parva) | Narrative | Task 3, 4 | https://sacred-texts.com/hin/m03/index.htm |

---

## Dataset statistics

| Task | Episodes | Domains covered | Difficulty |
|------|----------|-----------------|------------|
| Glossary Anchoring | 1500 | Ayurveda, Astronomy, Philosophy | Easy |
| Sandhi Resolution | 1500 | Philosophy, Ayurveda, Narrative | Medium |
| Samāsa Classification | 1500 | Philosophy, Narrative, Ayurveda, Astronomy | Medium |
| Referential Coherence | 1500 | Narrative, Philosophy | Hard |

---

## Baseline benchmark matrix

`baseline.py` now uses a single model selector env var: `BASELINE_MODEL`.
To compare multiple Cloudflare models, update `BASELINE_MODEL` and rerun baseline once per model.
If Cloudflare is rate-limited, the script automatically falls back to HF Router model
`Qwen/Qwen2.5-7B-Instruct` (from the curated free-tier list in `server/model_agent.py`).

Current recorded run from `baseline_results.json`:

| Provider | Model | Episodes | Seed | Glossary | Sandhi | Samasa | Coherence | Overall |
|----------|-------|----------|------|----------|--------|--------|-----------|---------|
| Cloudflare Workers AI | @cf/meta/llama-3.1-8b-instruct | 5 | 42 | 1.000 | 1.000 | 1.000 | 0.280 | 0.820 |
| Cloudflare Workers AI | @cf/meta/llama-3.1-70b-instruct | 5 | 42 | 1.000 | 1.000 | 1.000 | 0.700 | 0.925 |
| Cloudflare Workers AI | @cf/meta/llama-3.2-3b-instruct | 5 | 42 | 1.000 | 0.800 | 0.480 | 0.140 | 0.605 |
| Cloudflare Workers AI | @cf/meta/llama-3.3-70b-instruct-fp8-fast | 5 | 42 | 1.000 | 1.000 | 1.000 | 0.700 | 0.925 |
| Cloudflare Workers AI | @cf/meta/llama-3.3-70b-instruct-fp8-fast | 20 | 42 | 1.000 | 1.000 | 0.970 | 0.700 | 0.917 |

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

### Task 3 — Samāsa Classification

| Outcome | Reward |
|---------|--------|
| Correct compound type identified | `+1.00` |
| Adjacent type (e.g. Karmadharaya instead of Tatpurusha) | `+0.40` |
| Wrong compound type | `+0.00` |
| Invalid selection | `+0.00` + episode ends |

Six samāsa types are tested: **Tatpurusha**, **Karmadharaya**, **Dvigu**,
**Dvandva**, **Bahuvrihi**, and **Avyayibhava** — covering the full classical
Pāṇinian taxonomy. Each episode provides the compound in both IAST and
Devanagari alongside its source passage and English context.

### Task 4 — Referential Coherence

| Outcome | Reward |
|---------|--------|
| Each correct checkpoint answer | `+0.10` |
| Correct final antecedent identification | `+0.70` |
| Wrong checkpoint or final answer | `+0.00` |

All episode scores are normalized to **0.0–1.0** before being returned as the
final `reward` when `done=True`.

---

## Grader design — why no LLM, no BLEU

All four graders are fully deterministic:
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
- LLM providers' API key.
### Local development

```bash
# Clone
git clone https://huggingface.co/spaces/Adityahars/Sanskrit-env
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
│  rolling_memory = ""   ← starts empty each episode      │
│                                                         │
│  while not done:                                        │
│    1. THINK  — build prompt from obs + rolling_memory   │
│    2. ACT    — call Groq, get raw answer                │
│    3. MATCH  — match raw answer to candidate_options    │
│    4. STEP   — env.step(ManuscriptAction(selected))     │
│    5. UPDATE — append "Q → A" to rolling_memory         │
└─────────────────────────────────────────────────────────┘
```

The `rolling_memory` string grows by one line per step and is injected
into every prompt. For Task 4 this looks like:

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


## Contributing

Contributions welcome. Highest-priority areas:

1. **More episodes** — additional Sanskrit passages with annotated answers
   (must include IAST, Devanagari, English context, 4 options, correct answer)
2. **New domains** — Jyotisha (astrology), Natya Shastra (dramaturgy), Vedic hymns
3. **Harder sandhi cases** — especially involving anusvara, visarga, and vowel coalescence
4. **More samāsa episodes** — especially Dvigu and rarer Avyayibhava patterns
5. **Multi-language target** — currently English-only; Hindi or regional language target translations

Please open an issue before starting a large contribution.

---

## Citation

If you use SanskritEnv in your research:

```bibtex
@misc{sanskritenv2025,
  title   = {SanskritEnv: A Reinforcement Learning Environment for Sanskrit Manuscript Interpretation},
  author  = {Meta_Mesh},
  year    = {2026},
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
- [Sanskrit Sandhi Split Sighum](https://huggingface.co/datasets/chronbmm/sanskrit-sandhi-split-sighum) — annotated corpus reference
- [Itihasa](https://huggingface.co/datasets/rahular/itihasa) — annotated corpus reference
