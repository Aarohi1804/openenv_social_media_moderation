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

openenv
reinforcement-learning
social-media-content-moderation
misinformation
trust-and-safety
---


🛡️ Social Media Moderation Environment

A Reinforcement Learning training ground where AI learns to protect communities from misinformation — without silencing real voices.


The Problem This Solves
Every minute, 500 hours of content are uploaded to YouTube. Every day, 500 million tweets are posted. Buried inside that torrent are health hoaxes that kill people, political fabrications that destabilize democracies, and Coordinated Inauthentic Behavior (CIB) campaigns engineered specifically to evade automated detection.
Content moderation at this scale is impossible for humans alone. But training AI to moderate responsibly is harder than it looks. Every real Trust & Safety system faces the same trilemma:
ForceThe TensionSpeedViral misinformation spreads in minutes — decisions can't wait for fact-checkersAccuracySignals are noisy, ambiguous, and actively manipulated by adversariesFairnessOver-moderation suppresses free speech, erodes user trust, and creates legal liability
This environment operationalizes that trilemma as a learnable RL problem — forcing agents to internalize the same constraints that govern real production safety systems at platforms like Meta, YouTube, and Twitter/X.

This is not a fake-news detector. It is a training gym for moderation strategy.


Architecture Overview
Unlike standard RL environments that ship as simple Python libraries, this environment is built as a production-grade FastAPI application — designed from the ground up for horizontal scaling and multi-agent evaluation.
Application Stack
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
Infrastructure Choices
Modern Packaging (pyproject.toml, package = false)
The project deliberately moves away from legacy setuptools to a minimalist pyproject.toml with package = false. This treats the environment as a standalone deployable service rather than an installable library — the correct abstraction for a containerized RL server.
uv for Dependency Resolution
Dependencies are managed via astral-sh/uv, the Rust-based Python package manager. uv resolves and installs the full dependency graph significantly faster than pip, and the uv.lock file guarantees byte-for-byte reproducible builds across environments.
Multi-Stage Docker Build
The Dockerfile uses a two-stage build pattern:

Builder stage — installs uv, creates an isolated .venv, and resolves all dependencies into it via uv pip install --python .venv/bin/python
Final stage — copies only the compiled .venv and application code, producing a lightweight, production-hardened image with no build tooling included

The result is a container that builds cleanly, starts fast, and carries no unnecessary attack surface.

✨ Five Novel Design Choices
Each design decision reflects a real Trust & Safety engineering challenge — not a textbook RL trick.
1. Volume vs. Value Signal Separation (Adversarial Robustness)
Real Coordinated Inauthentic Behavior includes botnet-scale report brigading — flooding the reports queue on legitimate posts to trigger automated removal. Task 3 simulates this at realistic scale: a single brigaded real post can accumulate 1,000+ total reports, yet have only 0–2 from trusted reporters.
The observation space deliberately separates these signals:

report_count — raw volume, easily manipulated
trusted_report_count — credibility-weighted signal, harder to fake
reporter_trust — trust score of the single highest-credibility reporter

An agent that relies on report_count alone develops heuristic bias and will systematically over-moderate legitimate content under coordinated attack. An adversarially robust agent learns to weight trusted signal over raw volume.
[IMAGE: BOTNET_SPIKE] — Visual showing 1,000+ report spike vs. trusted_report_count=1 in Task 3 logs.
2. Asymmetric Penalties (Ethical Calibration)
Wrongly deleting a real post (−1.2) is penalized more harshly than missing a fake post (−1.0). This encodes a deliberate ethical position: censoring legitimate speech is not a neutral miss — it is an active harm with legal, reputational, and societal consequences.
Agents cannot score well by being systematically aggressive. They must internalize the asymmetry.
3. Delayed Ground Truth (Temporal Uncertainty)
factcheck_confidence starts at 0.1 and rises gradually across episode steps, simulating the real-world lag between content publication and fact-checker verdict. For coordinated campaign posts (Task 3), it never rises — modeling a breaking story in a domain with no established fact-checking coverage.
Agents must develop a risk-calibrated early-action strategy. Waiting for certainty is not an option.
4. Coordinated Campaign Detection (Multi-Step Reasoning)
In Task 3, five posts are secretly linked as part of a single CIB campaign. These posts share signal patterns and are permanently frozen at low factcheck_confidence. Per-post greedy strategies fail here — the agent must reason across the episode trajectory, not just react to each post in isolation.
This is the mechanic that separates Task 3 from a simply noisier version of Task 2.
5. ESCALATE Resolution Loop (Strategic Delay)
ESCALATE is not a cop-out — it is a genuine strategic instrument. The environment simulates human review with a realistic 2–3 step resolution delay: when a post is escalated, factcheck_confidence jumps to 1.0 after the delay, revealing ground truth. Escalating under genuine uncertainty (factcheck_confidence < 0.3) is rewarded; escalating when the answer is already clear (> 0.7) is penalized as lazy delegation.

🔭 Observation Space
The agent receives 9 signals per post. Ground truth (is_fake) is never exposed.
FieldTypeRangeDescriptionmisinfo_probabilityfloat0.0 – 1.0Classifier estimate of misinformation likelihood — noisy by designvirality_scorefloat0.0 – 1.0Current spread levelspread_velocityfloat0.0 – 1.0Rate at which virality is acceleratingreport_countint0 – 1,000+Raw report volume — inflatable by botnet brigadingtrusted_report_countint0 – 50Reports from high-credibility accounts onlyreporter_trustfloat0.0 – 1.0Trust score of the single highest-credibility reporteruser_credibilityfloat0.0 – 1.0Poster's historical trustworthinessis_repeat_offenderboolTrue/FalseWhether this account has been actioned beforefactcheck_confidencefloat0.0 – 1.0Fact-checker review progress — frozen for campaign postscontent_categorystrhealth / politics / entertainment / financeTopic — health and politics carry elevated harm weights

⚡ Action Space
Five actions spanning the full intervention spectrum, from least to most invasive:
ActionDescriptionALLOWLeave the post live with no interventionLABEL_WARNINGAttach a contextual warning labelREDUCE_REACHSuppress in the recommendation algorithmDELETERemove the post permanentlyESCALATEQueue for human review — resolves in 2–3 steps
This mirrors the actual intervention tiers used by production Trust & Safety systems. The agent must learn that the strongest action is not always the correct one.

🎯 Reward Function
Every reward value encodes a deliberate policy decision:
SituationActionRewardReasoningFake postDELETE+1.0Correct removalFake postLABEL_WARNING+0.6Proportionate — informs without silencingFake postREDUCE_REACH+0.5Cautious containmentFake post + viralALLOW−1.0Active harm in motionReal postALLOW+0.4Protecting free expressionReal postDELETE−1.2Over-censorship — steepest single-action penaltyRepeat offender + fakeDELETE+1.2Pattern recognition rewardedRepeat offender + fakeALLOW−1.5Agent has prior signal — inexcusable missRepeat offender + realDELETE−1.5Cannot assume guilt from history aloneESCALATE + factcheck < 0.3—+0.3Appropriate uncertainty managementESCALATE + factcheck > 0.7—−0.2Lazy delegation penalizedHealth/politics fake + viralALLOW−1.5Highest real-world harm categoryEarly action on accelerating poststrong action+0.3Proactive moderation bonusSame action 5+ times in a rowany−0.2Degenerate behaviour penaltyDELETE rate > 80% in episode—−0.5Over-moderation penalty

📊 Tasks & Graders
All graders are fully deterministic at seed=42, ensuring that RL agent comparisons are scientifically valid and reproducible across different model architectures and hardware.
Task 1 — Basic Moderation (Easy)

10 posts, 40% fake
High-confidence signals. No adversarial reporting. No repeat offenders. Normal fact-check cadence.
Grader: 0.6 × fake_control_rate + 0.4 × real_protection_rate
Smart agent: 0.85 – 1.0 | Random agent: ~0.0 – 0.2

Task 2 — Balancing Act (Medium)

15 posts, 35% fake
Signal overlap — fake posts can show misinfo_probability as low as 0.55. Adversarial reporting on ~20% of real posts. Some repeat offenders. Slower fact-check rise rate.
Grader: 0.5 × fake_control + 0.35 × real_protection + 0.15 × early_action_rate

Balance penalty (−0.2) applied when agents control fakes but destroy real posts equally


Smart agent: 0.65 – 0.85 | Random agent: ~0.3 – 0.5

Task 3 — Campaign Detection (Hard)

20 posts, 30% fake — including a 5-post Coordinated Inauthentic Behavior (CIB) campaign
Extreme signal noise. Botnet-scale brigading on ~40% of real posts (1,000+ report spikes). Campaign posts permanently frozen at factcheck_confidence = 0.1. Frequent repeat offenders.
Grader: 0.4 × campaign_detection + 0.35 × non_campaign_accuracy + 0.25 × harm_reduction

Heavy real-post protection penalty: −0.3 if protection rate < 40%


Smart agent: 0.50 – 0.75 | Baseline LLM score: 0.560 | Random agent: ~0.2 – 0.4


🚀 Official Demo Script
The inference.py file in the root directory is the official end-to-end evaluation script. It:

Initializes the environment locally
Connects an LLM agent via the OpenAI-compatible Groq API
Runs a full episode across all three tasks (task_easy, task_medium, task_hard)
Prints structured [START], [STEP], and [END] logs to stdout
Saves final scores to outputs/baseline_results.json

bash# Install dependencies
pip install -r requirements.txt

# Set API key (Groq — free, OpenAI-compatible)
export OPENAI_API_KEY="your_groq_api_key"

# Run
python inference.py

Quick Start (Connect to Live Environment)
pythonfrom client import SocialMediaModerationEnv
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

Running Locally
bash# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Health check
curl http://localhost:7860/health
# {"status": "healthy"}

📁 Project Structure
social_media_moderation_env/
├── models.py              # Pydantic models: ModerationAction, ModerationObservation
├── client.py              # WebSocket client for agent interaction
├── inference.py           # Official end-to-end demo + evaluation script
├── pyproject.toml         # Modern packaging config (package = false)
├── uv.lock                # Reproducible dependency lockfile (astral-sh/uv)
├── outputs/
│   └── baseline_results.json
├── openenv.yaml           # OpenEnv spec — 3 tasks, 3 graders
└── server/
    ├── app.py             # FastAPI application (all endpoints auto-generated)
    ├── Dockerfile         # Multi-stage build: builder + minimal final image
    ├── requirements.txt
    └── social_media_moderation_env_environment.py  # Core environment logic

🌍 From Simulator to Production
The simulator faithfully mirrors production moderation architecture. Signal sources change — the decision interface does not.
This SimulatorProduction Systemrandom.uniform() generates signalsTrained NLP/CV classifiers generate signalsSynthetic posts from parametric rulesBillions of real posts from live trafficInstant reward computationReward from real-world outcomes (days later)Single environment instanceHorizontally scaled fleet of parallel agents10–20 posts per episode~500 million posts per day (Twitter scale)Deterministic seed for reproducibilityA/B tested policy rollouts with statistical controls
Deployment path: (1) train on historical moderation decisions with known outcomes, (2) shadow mode alongside human moderators, (3) canary rollout to 1% of live traffic, (4) expand with continuous metric monitoring and policy drift detection.

🔬 Design Q&A
Why simulation over real data?
Real moderation datasets are privacy-sensitive, biased by prior policy decisions, and nearly impossible to label at scale with reliable ground truth. Simulation enables controlled experimentation, safe policy exploration, and fully reproducible benchmarking. The same agent architecture transfers directly to production when signal sources are swapped.
What prevents reward hacking?
Three independent mechanisms: (1) asymmetric penalties make systematic aggression or passivity costly, (2) the degenerate behaviour detector penalizes repetitive action sequences, (3) the over-deletion cap penalizes episode-level DELETE rates above 80%. An agent can only score well by genuinely discriminating between content types.
How does this handle adversarial robustness?
The separation of report_count from trusted_report_count is the core anti-manipulation mechanism. Volume-based signals are deliberately noiseable — agents that develop heuristic bias toward raw report counts will systematically fail under CIB conditions in Task 3.
Why are health and politics penalized more heavily?
Misinformation in these categories causes documented, measurable real-world harm: vaccine hesitancy driving preventable deaths, election interference affecting democratic outcomes, dangerous self-treatment based on fabricated medical claims. The reward function encodes this explicitly — health or politics fake posts allowed to go viral receive −1.5, the joint-highest penalty in the system.
