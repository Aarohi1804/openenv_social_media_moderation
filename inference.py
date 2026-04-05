import asyncio
import os
import json
import textwrap
import sys
from typing import List, Optional
from openai import OpenAI

# Ensure Python looks in the current directory for models and client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your actual environment and action models
from models import ModerationAction
from client import SocialMediaModerationEnv

# ─── MANDATORY CONFIGURATION ──────────────────────────────────────────────────
# Judges will inject these environment variables.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"

# Environment Metadata
TASK_NAME = os.getenv("MODERATION_TASK", "Task 3")
BENCHMARK = "social-media-moderation-env"

# Logic Constraints
MAX_STEPS = 20 
TEMPERATURE = 0.0 
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = """You are an AI content moderator.
You must respond with ONLY ONE of these words: ALLOW, LABEL_WARNING, REDUCE_REACH, DELETE, ESCALATE.

Rules:
1. If misinfo_probability > 0.65: DELETE.
2. If misinfo_probability 0.50-0.65: REDUCE_REACH.
3. If misinfo_probability < 0.50: ALLOW.
4. If factcheck_confidence < 0.3 AND unsure: ESCALATE.
No explanation. Just the action word."""

# ─── MANDATORY LOGGING FORMAT ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── AGENT LOGIC ──────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, obs) -> str:
    # Use model_dump_json() if using Pydantic v2, or json() if v1
    try:
        obs_text = f"Observation: {obs.model_dump_json()}"
    except AttributeError:
        obs_text = f"Observation: {obs.json()}"
        
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=TEMPERATURE,
            max_tokens=10,
        )
        action = completion.choices[0].message.content.strip().upper()
        # Fallback validation
        for valid in ["ALLOW", "LABEL_WARNING", "REDUCE_REACH", "DELETE", "ESCALATE"]:
            if valid in action: return valid
        return "ALLOW"
    except Exception:
        return "ALLOW"

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # ENV_URL for local testing. Judges will use the Space URL.
    ENV_URL = "http://localhost:8000" 
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    async with SocialMediaModerationEnv(base_url=ENV_URL) as env:
        try:
            result = await env.reset(task_id=3) 
            
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_str = get_model_action(client, result.observation)
                result = await env.step(ModerationAction(action=action_str))
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)

                if result.done:
                    break

            # Scoring calculation
            score = sum(rewards) / MAX_STEPS 
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

            # Save to outputs/ folder as required
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/baseline_results.json", "w") as f:
                json.dump({"score": score, "steps": steps_taken, "rewards": rewards}, f)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
    