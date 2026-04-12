---
title: Social Media Moderation Environment
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - social-media-content-moderation
  - misinformation
  - trust-and-safety
---

# 🛡️ Social Media Moderation Environment

> *A Trust & Safety training gym where AI agents learn to moderate social media content under adversarial conditions, information delay, and coordinated manipulation; without ever seeing ground truth.*

---

## 30-Second Executive Summary 

This is a **Partially Observable RL environment** that trains agents to make strategic content moderation decisions, the same decisions Trust & Safety teams at Meta, YouTube, and Twitter/X and other social media platforms, make billions of times per day.

**What makes it non-trivial:**
- Agents never see ground truth. They operate on noisy, gameable signals.
- Adversaries actively manipulate those signals (brigading, coordinated campaigns).
- Over-censorship is penalised *more harshly* than missing misinformation — encoding a real ethical trade-off.
- Task 3 requires multi-episode pattern recognition that simple threshold agents cannot solve.

**Three tasks. Three graders. One hard problem.**

---

## 🌍 The Problem This Solves

Every minute, 500 hours of content are uploaded to YouTube. Every day, 500 million tweets are posted. Buried inside that torrent are health hoaxes that lead to preventable deaths, and political fabrications engineered to spark riots, escalate geopolitical "cold wars," and destabilise democracies. 

The real-world consequences of viral misinformation are lethal. History has repeatedly shown how unmoderated public feeds can be weaponized—from viral posts inciting mob violence and city-wide riots, to the algorithmic amplification of ethnic conflict in crisis zones, to coordinated state-sponsored campaigns that manipulate global elections. 

Content moderation at this scale is impossible for humans alone. But training AI to moderate responsibly is harder than it looks. Every real Trust & Safety system faces the same trilemma:

| Force | The Tension |
|---|---|
| **Speed** | Viral misinformation spreads in minutes — decisions can't wait for fact-checkers. |
| **Accuracy** | Signals are noisy, ambiguous, and actively manipulated by adversaries. |
| **Fairness** | Over-moderation suppresses free speech, erodes user trust, and creates legal liability. |

This environment operationalises that trilemma as a learnable RL problem — forcing agents to internalise the same constraints that govern real production safety systems.

> *This is not a fake-news detector. It is a training gym for moderation strategy.*

## The Two Worlds (POMDP)

This is a **Partially Observable Markov Decision Process**. The hidden world and the observable world are deliberately separated — mirroring the information asymmetry that real moderation systems face.

| Hidden World (environment only) | Observable World (what agent sees) |
|---|---|
| `is_fake` — ground truth label | `misinfo_probability` — noisy estimate |
| `true_virality` — actual spread | `virality_score` — current spread level |
| `is_campaign` — part of coordinated attack | `factcheck_confidence` — rises slowly |
| `is_brigaded` — fake reports from bad actors | `report_count` vs `trusted_report_count` |

Real world moderation systems does not have access to ground truth in real time. It receives signals from ML models, reporters, and fact-checkers — all of which are imperfect and gameable. This environment models that reality exactly.

---

## Architecture Overview

```
Agent (any LLM / RL policy)
        │
        │  WebSocket (persistent, low-latency)
        ▼
FastAPI Application  ←  /reset  /step  /state  /health  /docs
        │
        ▼
SocialMediaModerationEnvironment
  ├── Post Generator (stochastic signal modelling)
  ├── Reward Engine (asymmetric, context-aware)
  ├── Escalation Resolver (2–3 step human review loop)
  └── Deterministic Grader (seed=42, scientifically reproducible)
```

**Modern packaging** — `pyproject.toml` with `package = false` treats the environment as a standalone deployable service, not an installable library. **`uv` for dependency resolution** — the `uv.lock` file guarantees byte-for-byte reproducible builds. **Multi-stage Docker build** — builder stage resolves dependencies; final stage ships only the compiled `.venv` and application code.

---

## ✨ Seven Novel Design Choices

Each design decision reflects a real Trust & Safety engineering challenge.

### 1. Volume vs. Value Signal Separation (Adversarial Robustness)

Real CIB includes **botnet-scale report brigading** — flooding the reports queue on legitimate posts to trigger automated removal. The observation space deliberately separates:

- `report_count` — raw volume, easily manipulated
- `trusted_report_count` — credibility-weighted, harder to fake
- `reporter_trust` — trust score of the single highest-credibility reporter

```
[TELEMETRY INTERCEPT — Task 3, Step 14]
post_id: #0xF3A2
  report_count:         187   ← botnet flood
  trusted_report_count:   1   ← only one legitimate reporter
  misinfo_probability:  0.31  ← classifier uncertain
  factcheck_confidence: 0.10  ← frozen (campaign post)

NAIVE AGENT   → DELETE  (fell for the flood)   reward: −1.2
TRAINED AGENT → ALLOW   (read the trusted signal)  reward: +0.4
```

An agent that relies on `report_count` alone will systematically over-moderate legitimate content under coordinated attack.

### 2. Asymmetric Penalties (Ethical Calibration)

Wrongly deleting a real post (`−1.2`) is penalised **more harshly** than missing a fake post (`−1.0`). This encodes a deliberate ethical position: censoring legitimate speech is not a neutral miss — it is an active harm with legal, reputational, and societal consequences. Agents cannot score well by being systematically aggressive.

### 3. Delayed Ground Truth (Temporal Uncertainty)

`factcheck_confidence` starts at `0.1` and rises gradually across episode steps. For **coordinated campaign posts**, it **never rises** — modelling a breaking story in a domain with no established fact-checking coverage. Agents must develop a risk-calibrated early-action strategy. Waiting for certainty is not an option.

### 4. Coordinated Campaign Detection (Multi-Step Reasoning)

In Task 3, five posts are secretly linked as part of a single CIB campaign. Their `factcheck_confidence` is permanently frozen. Per-post greedy strategies fail entirely — the agent must reason **across the episode trajectory**, not just react to each post in isolation. This is what separates Task 3 from a simply noisier version of Task 2.

### 5. ESCALATE Resolution Loop (Strategic Delay)

`ESCALATE` is not a cop-out — it is a genuine strategic instrument. The environment simulates human review with a 2–3 step resolution delay: `factcheck_confidence` jumps to `1.0` for the escalated post after the delay. Escalating under genuine uncertainty (`factcheck_confidence < 0.3`) is rewarded; escalating when the answer is already clear (`> 0.7`) is penalised as lazy delegation.

### 6. Systemic Safeguards (Degeneracy & Over-Censorship Caps)
Most RL environments allow agents to "hack" the reward function by finding a single safe action and spamming it. This environment introduces two hard systemic penalties:
* **Degenerate Behavior Detector:** Applying the exact same action 5+ times in a row incurs a continuous `−0.2` penalty, forcing the agent to actively evaluate each post rather than settling into a passive local optima.
* **Mass-Censorship Cap:** If an agent deletes more than 80% of the total posts in an episode, it triggers a massive `−0.5` end-of-episode penalty. Brute-force censorship is mathematically disqualified.

  ### 7. Phase-2 Validated Epsilon Clamping
Standard RL environments often crash downstream evaluation pipelines when scores hit absolute `0.0` or `1.0` due to division-by-zero errors in automated orchestrators. This environment implements strict Epsilon Clamping `(0.001 < score < 0.999)` on all final grader outputs. It guarantees mathematical stability and ensures the environment can be safely deployed in horizontally scaled, automated evaluation pipelines without crashing the parent container.

---

## 🔭 Observation Space

The agent receives **10 observable signals** per post, **plus 2 episode-progress fields**. Ground truth (`is_fake`) is never exposed.

| Field | Type | Range | Description |
|---|---|---|---|
| `misinfo_probability` | float | 0.0 – 1.0 | Estimated probability of misinformation. Overlaps between fake and real posts to force uncertainty. |
| `virality_score` | float | 0.0 – 1.0 | Current spread level of the post. |
| `spread_velocity` | float | 0.0 – 1.0 | Rate at which virality is **accelerating**. Enables early intervention before full virality. |
| `report_count` | int | 0 – 200 | Total user reports. **Can be inflated by coordinated brigading.** |
| `trusted_report_count` | int | 0 – 50 | Reports from verified high-credibility users only (trust > 0.7). |
| `reporter_trust` | float | 0.0 – 1.0 | Trust score of the single highest-credibility reporter who flagged this post. |
| `user_credibility` | float | 0.0 – 1.0 | Historical integrity score of the posting account. |
| `is_repeat_offender` | bool | True/False | Flag for accounts with confirmed prior policy violations. |
| `factcheck_confidence` | float | 0.0 – 1.0 | Human-in-the-loop audit progress. Starts at 0.1, rises over time. **Frozen for campaign posts.** |
| `content_category` | string | health / politics / entertainment / finance | Topic-based risk weighting. Health and politics carry higher penalties. |
| `step_number` | int | — | Current step in the episode. |
| `posts_remaining` | int | — | Posts remaining in this episode. |

### Why the reporting split matters

A naive design uses a single `reporter_trust` average. But in reality, brigading makes averages deeply misleading. By exposing both `report_count` (total, gameable) and `trusted_report_count` (credible only), the agent must learn to reason: *"150 reports but only 1 trusted reporter → likely a brigade, do not over-react."* This is exactly how real Trust & Safety systems handle coordinated manipulation.

---

## ⚡ Action Space

Five actions spanning the full intervention spectrum — mirroring the actual tiers used by production Trust & Safety platforms:

| Action | Effect | When to use |
|---|---|---|
| `ALLOW` | Post stays up unchanged | Confident the post is real |
| `LABEL_WARNING` | Attaches a contextual warning label | Fake but low virality, or borderline |
| `REDUCE_REACH` | Suppresses in recommendation algorithm | Fake and spreading, but uncertain |
| `DELETE` | Removes the post permanently | Confident fake, especially viral or repeat offender |
| `ESCALATE` | Queues for human review — resolves in 2–3 steps | Genuinely uncertain, factcheck still low |

**The agent must learn that the strongest action is not always the correct one.**

---

## 🎯 Reward Function

Every value encodes a deliberate policy decision. The core trade-off: **over-censorship is penalised more harshly than under-moderation.**

To ensure mathematical stability for the RL agent and prevent division-by-zero errors in downstream evaluation, rewards are first calculated as a `raw_reward`. Before returning to the agent, the environment applies a strict **Epsilon-Bounded Normalization Formula** to keep the final output safely between `(0, 1)`:
`normalized_reward = max(0.01, min(0.99, (raw_reward + 2.5) / 5.0))`

### 1. Base Action Rewards (Raw)
| Ground Truth | Action | Raw Reward | Reasoning |
|---|---|---|---|
| 🔴 Fake | `DELETE` | `+1.0` (+ `virality * 0.5`) | Correct removal. Reward dynamically scales up if the post is highly viral. |
| 🔴 Fake | `ALLOW` | `−1.0` | Active harm in motion — inexcusable miss. |
| 🔴 Fake | `LABEL` / `REDUCE` / `ESCALATE` | `+0.5` | Partial containment or safe delegation. Better than allowing, but weaker than deletion. |
| 🟢 Real | `ALLOW` | `+0.4` | Protecting free expression. |
| 🟢 Real | `DELETE` | **`−1.2`** | Over-censorship — the steepest single-action base penalty in the system. |
| 🟢 Real | `LABEL` / `REDUCE` / `ESCALATE` | `−0.2` | Minor over-action / unnecessary friction on legitimate speech. |

### 2. Contextual & Systemic Modifiers
These modifiers dynamically adjust the `raw_reward` based on episode history and metadata before normalization occurs.

| Trigger | Effect on Reward | Purpose |
|---|---|---|
| **Repeat Offender (Fake Post)** | `DELETE` base becomes `+1.2`<br>`ALLOW` drops to `−1.5` | Amplifies the requirement to recognize and strictly moderate known bad actors. |
| **Repeat Offender (Real Post)** | `DELETE` drops to `−1.5` | Penalises assuming guilt based purely on account history without evaluating the specific post. |
| **Degeneracy Prevention** | `−0.2` per step | Triggers if the agent applies the exact same action 5+ times in a row. Forces active evaluation. |
| **Over-Moderation Cap** | `−0.5` at episode end | Triggers if the agent deletes > 80% of all posts in the episode. Disqualifies brute-force censorship. |
---

## 📊 Tasks and Graders

All graders are **fully deterministic at `seed=42`**, ensuring scientifically valid and reproducible RL agent comparisons.

### Task 1 — Basic Moderation *(Easy)*

**10 posts | 40% fake**

Clear, unambiguous signals. `misinfo_probability` for fake posts: 0.75–0.95. No brigading, no repeat offenders, no campaigns. Normal fact-check cadence.

**Objective:** Basic signal literacy — action fake posts, protect real ones.

```
score = (0.6 × fake_control_rate) + (0.4 × real_protection_rate) − over_deletion_penalty
```

*Expected: Smart agent 0.85–1.0 | Random agent ~0.0–0.2*

---

### Task 2 — Balancing Act *(Medium)*

**15 posts | 35% fake**

Signals overlap — fake posts can show `misinfo_probability` as low as 0.55. Brigading on ~20% of real posts. Some repeat offenders. Slower fact-check rise.

**Objective:** Balance fake containment against over-moderation of suspicious-looking real posts. Reward early intervention.

```
score = (0.50 × fake_control) + (0.35 × real_protection) + (0.15 × early_action_rate) − balance_penalty
```

Balance penalty (−0.2) when agents control fakes but equally destroy real posts.

*Expected: Smart agent 0.65–0.85 | Random agent ~0.3–0.5*

---

### Task 3 — Campaign Detection *(Hard)*

**20 posts | 30% fake — including a 5-post CIB campaign**

Extreme signal noise. Botnet-scale brigading on ~40% of real posts. Campaign posts permanently frozen at `factcheck_confidence = 0.1`. Frequent repeat offenders.

**Why this is genuinely hard for frontier LLMs:** A simple threshold agent cannot distinguish campaign posts from organic fake posts. The agent must track the frozen factcheck pattern across the full episode trajectory — per-post greedy strategies fail entirely.

**Objective:** Detect the CIB campaign while handling noisy signals, brigading, and repeat offenders simultaneously.

```
score = (0.40 × campaign_detection) + (0.35 × non_campaign_accuracy) + (0.25 × harm_reduction) − protection_penalty
```

Heavy real-post protection penalty: −0.3 if protection < 40%, −0.15 if protection < 60%.

*Expected: Smart agent 0.50–0.75 | Random agent ~0.2–0.4*

---

## 📈 Baseline Results

Evaluated using `llama-3.3-70b-versatile` via Groq API. Scores represent programmatic grader assessment of full episodes at `seed=42`.

| Task | LLM Score | Total Reward | Steps |
|---|---|---|---|
| Task 1 — Basic Moderation | 0.933 | 6.1 | 10 |
| Task 2 — Balancing Act | 0.740 | 7.2 | 15 |
| Task 3 — Campaign Detection | 0.920 | 10.1 | 20 |
| **Average** | **0.864** | — | — |

**LLM vs Rule-Based comparison** (rule-based: `misinfo_probability > 0.6 → DELETE`):

| Task | LLM Agent | Rule-Based | Verdict |
|---|---|---|---|
| Task 1 | 0.933 | 1.0 | Rule-based wins on simple signals |
| Task 2 | 0.740 | 0.81 | LLM confused by brigaded real posts |
| Task 3 | **0.920** | 0.872 | **LLM wins** — multi-signal reasoning detects campaign |

The LLM outperforms the rule-based agent on Task 3 — which is the point. Simple threshold logic cannot detect coordinated campaigns. A richer reasoning agent is required, and this environment rewards it.

---
## 📁 Project Structure

```text
social_media_moderation_env/
├── inference.py           ← Official end-to-end evaluation script (required)
├── models.py              ← Pydantic models: ModerationAction, ModerationObservation
├── client.py              ← WebSocket client for agent interaction
├── openenv.yaml           ← OpenEnv spec — 3 tasks, 3 graders
├── pyproject.toml         ← Modern packaging config (package = false)
├── requirements.txt       ← Standard Python dependency list
├── uv.lock                ← Reproducible dependency lockfile
├── Dockerfile             ← Container definition
├── outputs/
│   └── inference_results.json
└── server/
    ├── app.py             ← FastAPI application (all endpoints auto-generated)
    ├── __init__.py
    └── social_media_moderation_env_environment.py
---

## 🚀 Quick Start

**Connect to the live environment:**

```python
from client import SocialMediaModerationEnv
from models import ModerationAction

with SocialMediaModerationEnv(
    base_url="https://aarohiii-social-media-moderation-env.hf.space"
) as env:
    obs = env.reset(task_id=3)
    print(f"misinfo_probability: {obs.misinfo_probability}")
    print(f"factcheck_confidence: {obs.factcheck_confidence}")
    print(f"trusted_report_count: {obs.trusted_report_count}")

    result = env.step(ModerationAction(action="REDUCE_REACH"))
    print(f"Reward: {result.reward}")
```

**Run the inference script:**

```bash
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

**Run locally:**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
curl http://localhost:7860/health
# {"status": "healthy"}
```

---

## 🌍 From Simulator to Production

The simulator faithfully mirrors production moderation architecture. Signal sources change — the decision interface does not.

| This Simulator | Real Production System |
|---|---|
| `random.uniform()` generates signals | Trained NLP + CV classifiers generate signals |
| Synthetic posts from parametric rules | Billions of real posts from live traffic |
| Instant reward computation | Reward from real-world outcomes, days later |
| Single environment instance | Horizontally scaled fleet of parallel agents |
| 10–20 posts per episode | ~500 million posts per day (Twitter scale) |
| Deterministic seed for reproducibility | A/B tested policy rollouts with statistical controls |

**Deployment path:** (1) train on historical moderation decisions with known outcomes, (2) shadow mode alongside human moderators, (3) canary rollout to 1% of live traffic, (4) expand with continuous metric monitoring and policy drift detection.

---

## 🔬 Q&A for Judges

**Q: Why simulation over real data?**
Real moderation datasets are privacy-sensitive, biased by prior policy decisions, and nearly impossible to label at scale with reliable ground truth. Simulation enables controlled experimentation, safe policy exploration, and fully reproducible benchmarking. The same agent architecture transfers directly to production when signal sources are swapped.

**Q: What prevents reward gaming?**
Three independent mechanisms: (1) asymmetric penalties make systematic aggression or passivity costly, (2) the degenerate behaviour detector penalises repetitive action sequences (same action 5+ times → −0.2), (3) the over-deletion cap penalises episode-level DELETE rates above 80%. An agent can only score well by genuinely discriminating between content types.

**Q: How does this handle adversarial robustness?**
The separation of `report_count` from `trusted_report_count` is the core anti-manipulation mechanism. Volume-based signals are deliberately noiseable — agents that develop heuristic bias toward raw report counts will systematically fail under CIB conditions in Task 3.

**Q: What makes Task 3 hard enough for frontier LLMs?**
Campaign posts have frozen `factcheck_confidence` — it never rises no matter how many steps pass. A simple threshold agent cannot detect this. The agent must track this temporal pattern across the full episode. Rule-based agents score ~0.87 on Task 3; the LLM scores 0.92 specifically because it reasons across multiple signals simultaneously.

**Q: Why are health and politics penalised more heavily?**
Misinformation in these categories causes documented, measurable real-world harm: vaccine hesitancy driving preventable deaths, election interference affecting democratic outcomes, dangerous self-treatment based on fabricated medical claims. The reward function encodes this explicitly — health or politics fake posts allowed to go viral receive −1.5, the joint-highest penalty in the system.

**Q: How does this scale?**
The environment defines the decision interface. In production, the same interface connects to live ML models rather than stochastic signal generators. The policy learned here transfers directly — only the signal source changes.

---


