import asyncio
import os
import json
import sys
from typing import List, Optional
from openai import OpenAI

# making sure the environment module is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModerationAction
from client import SocialMediaModerationEnv

# ─── MODERATOR MANDATORY CONFIGURATION ──────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy_key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860") # Using 7860 to match your app_port
BENCHMARK = "social-media-moderation-env"

# Logic Constraints
MAX_STEPS = 20
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an AI content moderator.
Respond with ONLY ONE of these words: ALLOW, LABEL_WARNING, REDUCE_REACH, DELETE, ESCALATE.
Rules:
1. If misinfo_probability > 0.65: DELETE.
2. If misinfo_probability 0.50-0.65: REDUCE_REACH.
3. If misinfo_probability < 0.50: ALLOW.
4. If factcheck_confidence < 0.3 AND unsure: ESCALATE."""

# ─── MANDATORY LOGGING FORMAT (FLUSH=TRUE) ──────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = f'"{error}"' if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float], task: str) -> None:
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    # MODERATOR FIX: Adding task to the END log just to be super safe
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards=[{rewards_str}]", flush=True)

# ─── AGENT LOGIC ────────────────────────────────────────────────
def get_model_action(client: OpenAI, obs) -> str:
    try:
        obs_text = f"Observation: {obs.model_dump_json()}" if hasattr(obs, 'model_dump_json') else f"Observation: {obs.json()}"
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text}
            ],
            temperature=TEMPERATURE,
            max_tokens=10,
        )
        action = completion.choices[0].message.content.strip().upper()
        
        for valid in ["ALLOW", "LABEL_WARNING", "REDUCE_REACH", "DELETE", "ESCALATE"]:
            if valid in action: return valid
            
        return "ALLOW"
    except Exception:
        return "ALLOW"

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Strict testing of all 3 tasks with exact logging format and clamping logic
    tasks_to_test = ["task_easy", "task_medium", "task_hard"]
    
    async with SocialMediaModerationEnv(base_url=ENV_URL) as env:
        for current_task in tasks_to_test:
            rewards = []
            steps_taken = 0
            score = 0.01
            success = False
            
            # 1. Print exact [START] log
            log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)
            
            try:
                result = await env.reset(task_id=current_task)
                
                for step in range(1, MAX_STEPS + 1):
                    if result.done: break
                    
                    action_str = get_model_action(client, result.observation)
                    result = await env.step(ModerationAction(action=action_str))
                    
                    reward = result.reward or 0.0
                    rewards.append(reward)
                    steps_taken = step
                    
                    log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)
                    
                # 2. STRICT CLAMP CALCULATION 
                if rewards:
                    raw_score = sum(rewards) / len(rewards)
                    score = float(max(0.01, min(0.99, raw_score)))
                else:
                    score = 0.01
                    
                success = score >= SUCCESS_SCORE_THRESHOLD
                
            except Exception as e:
                print(f"Error during inference: {e}")
                score = 0.01 # Fallback for safety
            finally:
                # 3. Print exact [END] log
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards, task=current_task)

if __name__ == "__main__":
    asyncio.run(main())