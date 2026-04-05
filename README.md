---
title: Social Media Moderation Environment
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - social-media-content-moderation
  - misinformation
  - trust-and-safety
---

# 🛡️ Social Media Moderation Environment

> *A Reinforcement Learning training ground where AI learns to protect communities from misinformation — without silencing real voices.*

---

## The Problem This Solves

Every minute, **500 hours of content** are uploaded to YouTube. Every day, **500 million tweets** are posted. Buried inside that torrent are health hoaxes that kill people, political fabrications that destabilize democracies, and coordinated disinformation campaigns designed to evade detection.

Content moderation at this scale is **impossible for humans alone**. But training AI to moderate responsibly is harder than it sounds:

- **Misinformation is ambiguous** — a post about a new drug treatment might be genuine medical advice or dangerous pseudoscience. Signals overlap.
- **Bad actors adapt** — when they can't spread a fake post organically, they mass-report *real* posts to get them removed instead (brigading).
- **Over-moderation is also harmful** — wrongly silencing a real post suppresses free speech, erodes user trust, and exposes platforms to regulatory risk.
- **Fact-checkers are slow** — by the time the verdict arrives, the lie may have already gone viral.

Real platforms like Meta, Twitter/X, and YouTube face this exact trilemma daily: **speed vs. accuracy vs. fairness**. This environment operationalizes that trilemma as a learnable RL problem.

---

## What This Environment Is

A **Partially Observable Markov Decision Process (POMDP)** that simulates a content moderation system. An AI agent acts as a moderator — receiving noisy signals about incoming social media posts and making strategic decisions about each one.

The agent **never sees the ground truth.** It only sees what a real production system would see: imperfect classifier outputs, report patterns, credibility signals, and fact-checker confidence scores that arrive with delay. It must learn to act wisely under genuine uncertainty.

> *This is not a fake-news detector. It is a training gym for moderation strategy.*

---

## ✨ Five Novel Design Choices

This environment was designed to prevent the failure modes common in naive RL benchmarks. Each novelty reflects a real Trust & Safety challenge.

### 1. Adversarial Reporting (Brigading)
Real posts can be mass-reported by coordinated bad actors to trick the system into removing them. The environment simulates this: some legitimate posts receive 150+ total reports, but only 0–2 from *trusted* reporters. A naive agent that relies on raw `report_count` will over-moderate real content. A smart agent learns to weight `trusted_report_count` and `reporter_trust` instead.

### 2. Asymmetric Penalties (Safety-First Ethics)
Wrongly deleting a real post (`−1.2`) is penalized **more harshly** than missing a fake post (`−1.0`). This reflects the real ethical calculus: censoring legitimate speech is a serious harm, not just a neutral miss. Agents must internalize this asymmetry to score well.

### 3. Delayed Ground Truth
`factcheck_confidence` starts at `0.1` at episode start and rises gradually as the episode progresses. For coordinated campaign posts (Task 3), it **never rises** — simulating a breaking story that no fact-checker has yet covered. Agents cannot wait for certainty; they must develop a risk-calibrated early-action strategy.

### 4. Coordinated Campaign Detection (Multi-Step Reasoning)
In Task 3, five posts are secretly part of a single coordinated misinformation campaign. These posts have related signals and never receive independent fact-checking. The agent must reason *across posts* — not just evaluate each one in isolation — to detect the campaign and respond proportionally. Per-post greedy strategies fail here.

### 5. ESCALATE Resolution Loop
Choosing `ESCALATE` is not a cop-out — it's a genuine strategic option. The environment simulates human review: after 2–3 steps, `factcheck_confidence` jumps to `1.0` for the escalated post, revealing the truth. Escalating under low confidence (`< 0.3`) is rewarded; escalating when the answer is obvious (`> 0.7`) is penalized as laziness.

---

## 🔭 Observation Space

The agent receives 9 signals per post. It never sees `is_fake`.

| Field | Type | Range | Description |
|---|---|---|---|
| `misinfo_probability` | float | 0.0 – 1.0 | Classifier estimate of misinformation likelihood — *noisy* |
| `virality_score` | float | 0.0 – 1.0 | Current spread level |
| `spread_velocity` | float | 0.0 – 1.0 | Rate at which virality is accelerating |
| `report_count` | int | 0 – 200 | Total user reports — *can be inflated by brigading* |
| `trusted_report_count` | int | 0 – 50 | Reports from high-credibility users only |
| `reporter_trust` | float | 0.0 – 1.0 | Trust score of the single highest-credibility reporter |
| `user_credibility` | float | 0.0 – 1.0 | Historical trustworthiness of the poster |
| `is_repeat_offender` | bool | True/False | Whether this user has been actioned before |
| `factcheck_confidence` | float | 0.0 – 1.0 | How much human fact-checkers have reviewed this post |
| `content_category` | str | health/politics/entertainment/finance | Topic — health and politics carry higher risk weights |

**Key insight:** `report_count` and `trusted_report_count` are split intentionally. A brigaded real post has `report_count: 180, trusted_report_count: 1`. A genuinely dangerous fake post has `report_count: 85, trusted_report_count: 9`. The agent must learn to read the difference.

---

## ⚡ Action Space

Five actions, ordered from least to most intervention:

| Action | Description |
|---|---|
| `ALLOW` | Leave the post live with no intervention |
| `LABEL_WARNING` | Add a contextual warning label |
| `REDUCE_REACH` | Suppress in the recommendation algorithm |
| `DELETE` | Remove the post completely |
| `ESCALATE` | Queue for human review — resolves in 2–3 steps |

This mirrors the actual intervention spectrum used by major platforms. The agent must learn that **the right action is not always the strongest one.**

---

## 🎯 Reward Function

The reward function encodes the ethical priorities of responsible moderation:

| Situation | Action | Reward | Reasoning |
|---|---|---|---|
| Fake post | `DELETE` | +1.0 | Correct removal |
| Fake post | `LABEL_WARNING` | +0.6 | Reasonable, proportionate |
| Fake post | `REDUCE_REACH` | +0.5 | Cautious — good but limited |
| Fake post + viral | `ALLOW` | −1.0 | Harm already in motion |
| Real post | `ALLOW` | +0.4 | Protecting free expression |
| Real post | `DELETE` | **−1.2** | Over-censorship — hardest penalty |
| Repeat offender + fake | `DELETE` | +1.2 | Pattern recognition rewarded |
| Repeat offender + fake | `ALLOW` | −1.5 | Agent should know better |
| Repeat offender + real | `DELETE` | −1.5 | Cannot assume guilt by history |
| `ESCALATE` + factcheck < 0.3 | — | +0.3 | Smart uncertainty management |
| `ESCALATE` + factcheck > 0.7 | — | −0.2 | Laziness penalized |
| Health/politics fake + viral | `ALLOW` | −1.5 | Highest real-world harm category |
| Early action on accelerating post | strong action | +0.3 | Proactive moderation bonus |
| Same action 5+ times in a row | any | −0.2 | Degenerate behaviour penalty |
| DELETE rate > 80% in episode | — | −0.5 | Over-moderation penalty |

---

## 📊 Tasks & Graders

All graders return a score in `[0.0, 1.0]` and are fully deterministic at `seed=42`.

### Task 1 — Basic Moderation *(Easy)*
- **10 posts, 40% fake**
- Clear, high-confidence signals. No adversarial reporting. No repeat offenders.
- **Grader:** `0.6 × fake_control_rate + 0.4 × real_protection_rate`
- *Expected smart agent score: 0.85 – 1.0 | Random agent: ~0.0 – 0.2*

### Task 2 — Balancing Act *(Medium)*
- **15 posts, 35% fake**
- Signal overlap (fake posts can show `misinfo_probability` as low as 0.55). Adversarial reporting on ~20% of real posts. Some repeat offenders. Slower fact-check rise.
- **Grader:** `0.5 × fake_control + 0.35 × real_protection + 0.15 × early_action_rate`
  - Penalizes agents that control fakes but destroy real posts (`−0.2` balance penalty)
- *Expected smart agent score: 0.65 – 0.85 | Random agent: ~0.3 – 0.5*

### Task 3 — Campaign Detection *(Hard)*
- **20 posts, 30% fake** — including a **5-post coordinated misinformation campaign**
- Extreme signal noise (`misinfo_probability` as low as 0.45 for fake posts). Frequent brigading (~40% of real posts). Campaign posts have `factcheck_confidence` permanently frozen at 0.1.
- **Grader:** `0.4 × campaign_detection + 0.35 × non_campaign_accuracy + 0.25 × harm_reduction`
  - Heavy penalty if real post protection < 40% (`−0.3`)
- *Expected smart agent score: 0.50 – 0.75 (Baseline LLM Score: 0.560) | Random agent: ~0.2 – 0.4*

---

## 🚀 Quick Start

```python
from client import SocialMediaModerationEnv
from models import ModerationAction

# Connect to the live environment
with SocialMediaModerationEnv(base_url="https://huggingface.co/spaces/aarohiii/social-media-moderation-env") as env:
    # Start Task 3 (Hard mode)
    obs = env.reset(task_id=3)
    print(f"Post misinfo probability: {obs.misinfo_probability}")
    print(f"Fact-check confidence: {obs.factcheck_confidence}")

    # Take a moderation action
    result = env.step(ModerationAction(action="REDUCE_REACH"))
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
```

---

## 💻 Running Locally

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic numpy openai

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Verify it's running
curl http://localhost:8000/health
# {"status": "healthy"}
```

## Running the Baseline Agent

```bash
# Uses Groq's free API (OpenAI-compatible format)
export OPENAI_API_KEY="your_groq_api_key"
python inference.py
```

The script outputs structured `[START]`, `[STEP]`, and `[END]` logs and saves final scores to `outputs/baseline_results.json`.

## Deploying to Hugging Face Spaces

```bash
openenv push --repo-id your-username/social-media-moderation-env
```

---

## 📁 Project Structure

```
social_media_moderation_env/
├── models.py              # Pydantic models: ModerationAction, ModerationObservation
├── client.py              # WebSocket client for agent interaction
├── inference.py           # Baseline LLM evaluation script (OpenAI API format)
├── outputs/               # baseline_results.json saved here
├── openenv.yaml           # OpenEnv spec metadata
└── server/
    ├── app.py             # FastAPI server (reset, step, state, health, docs)
    ├── Dockerfile         # Container definition
    └── social_media_moderation_env_environment.py  # Core environment logic
```

---

## 🌍 From Simulator to Production

This simulator faithfully mirrors the architecture of real production moderation systems. The only differences are the signal sources:

| This Simulator | Real Production System |
|---|---|
| `random.uniform()` generates signals | Trained NLP/CV models generate signals |
| Synthetic posts from parametric rules | Billions of real posts from live users |
| Instant reward computation | Reward derived from real-world outcomes (days later) |
| One environment instance | Thousands of parallel agents |
| 10–20 posts per episode | ~500 million posts per day (Twitter scale) |

The path to deployment: (1) train on historical moderation decisions with known outcomes, (2) run in shadow mode alongside human moderators, (3) canary rollout to 1% of live traffic, (4) expand with continuous monitoring.

---

## 🔬 Design Q&A

**Why simulation instead of real data?**
Real moderation data is privacy-sensitive, biased by prior policy decisions, and difficult to label ground truth for. Simulation enables controlled experimentation, safe policy exploration, and fully reproducible benchmarking — without any of those constraints.

**What prevents the agent from gaming the reward?**
Three mechanisms: (1) the asymmetric penalty structure makes it costly to be systematically aggressive or passive, (2) the degenerate behaviour detector penalizes repetitive action sequences, (3) the over-deletion penalty caps episode-level DELETE rates at 80%. An agent can only score well by genuinely discriminating between fake and real content.

**How does this scale?**
The simulator defines the decision interface. In production, simulated signals are replaced by live model outputs — the same trained policy can operate on real data without architectural changes. This is the standard research → deployment pipeline used at major platforms.

**Why are health and politics weighted more heavily?**
Misinformation in these categories causes measurable real-world harm — vaccine hesitancy, election interference, medical self-treatment based on false information. The reward function encodes this: a health/politics fake post allowed to go viral receives `−1.5`, the harshest penalty in the system.

---


