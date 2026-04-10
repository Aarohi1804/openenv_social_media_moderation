import asyncio
import os
import json
import sys
import re # Added for regex validation of logs
from typing import List, Optional
from openai import OpenAI

# Ensure Python looks in the current directory for models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModerationAction
from client import SocialMediaModerationEnv

# ─── STRICT RULE COMPLIANCE: ENVIRONMENT VARIABLES ─────────────────────────────
# MUST match the exact logic from the email guidelines
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000") 
BENCHMARK = "social-media-moderation-env"

MAX_STEPS = 20 
TEMPERATURE = 0.0 
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = """You are a Senior Trust & Safety AI Moderator.
Your job is to analyze social media observations for misinformation, bot campaigns, and policy violations.

First, write a 1-2 sentence internal thought process analyzing the nuance, user history, and potential harm of the observation.
Then, on a new line, output your final decision wrapped in exact brackets.

Valid decisions: [ALLOW], [LABEL_WARNING], [REDUCE_REACH], [DELETE], [ESCALATE].

Example:
Thought: The misinformation probability is 0.68, which crosses the strict threshold. The user has a low trust score, indicating a potential coordinated bot network.
Decision: [DELETE]"""

# ─── STRICT RULE COMPLIANCE: LOGGING FORMAT ───────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # REMOVED task and score variables to perfectly match the required regex
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs) -> str:
    try:
        obs_text = f"Observation: {obs.model_dump_json()}" if hasattr(obs, 'model_dump_json') else f"Observation: {obs.json()}"
    except Exception:
        obs_text = str(obs)
        
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1, # Slight bump for reasoning variance
            max_tokens=150,  # CRITICAL: Given room to think
        )
        raw_output = completion.choices[0].message.content.strip()
        
        # The Regex Sniper: Looks specifically for a valid action wrapped in brackets
        match = re.search(r'\[(ALLOW|LABEL_WARNING|REDUCE_REACH|DELETE|ESCALATE)\]', raw_output, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()
            
        # If the LLM goes rogue and doesn't use brackets, default to safety
        return "ESCALATE" 
        
    except Exception as e:
        # Real enterprise systems fail safely
        return "ESCALATE"

async def main() -> None:
    # Initialize client exactly as requested
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    tasks_to_test = ["task_easy", "task_medium", "task_hard"]
    
    async with SocialMediaModerationEnv(base_url=ENV_URL) as env:
        for current_task in tasks_to_test:
            rewards = []
            steps_taken = 0
            score = 0.01
            success = False
            
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

                if rewards:
                    raw_score = sum(rewards) / len(rewards)
                    score = float(max(0.01, min(0.99, raw_score))) 
                else:
                    score = 0.01 
                
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception as e:
                # Do not print random python errors to stdout, it might break the bot's log parser
                pass
            finally:
                # Log end with EXACT formatting
                log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())