import asyncio
import os
import json
import sys
import re 
import time # Added for Rate Limiting
from typing import List, Optional
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModerationAction
from client import SocialMediaModerationEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Set to 7860 directly as you fixed
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860") 
BENCHMARK = "social-media-moderation-env"

MAX_STEPS = 20 
TEMPERATURE = 0.1 
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = """You are an Elite Meta Trust & Safety AI.

### STRICT MODERATION PROTOCOLS:
1. THE KILL SWITCH: If 'ENVIRONMENTAL WARNING' contains the word 'CRITICAL', you MUST output [DELETE]. System alerts override everything.
2. THE NOISE TRAP (Crucial): 'misinfo_probability' is heavily NOISY and will lie to you to test your logic! Do NOT blindly trust it. 
   - If misinfo_probability is low (< 0.50) BUT user_credibility is also low (< 0.40) or dynamic_reputation < 0.60, it is a disguised fake post. Output [REDUCE_REACH] or [DELETE].
   - If misinfo_probability is high (> 0.70) BUT user_credibility is high (> 0.80), it might be a false alarm. Output [REDUCE_REACH] or [LABEL_WARNING].
3. REPUTATION PENALTY: If user 'dynamic_reputation' < 0.45, be highly aggressive. Use [DELETE].
4. DEFAULT SAFE: Only output [ALLOW] if misinfo_probability is low AND user_credibility is high (> 0.60).

You must output ONLY the action name in brackets. Do not include any reasoning or extra text.
Example: [REDUCE_REACH]"""
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs) -> str:
    try:
        if isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
            
        warning_status = obs_dict.get('environmental_warning', 'None')
        user_rep = obs_dict.get('dynamic_reputation', 0.5)
        obs_json = json.dumps(obs_dict)
        
        obs_text = f"""!!! CRITICAL THREAT INTEL !!!
ENVIRONMENTAL WARNING: {warning_status}
CURRENT USER REPUTATION: {user_rep}

FULL OBSERVATION DATA:
{obs_json}"""

    except Exception as e:
        print(f"🚨 OBS PARSE ERROR: {e}", file=sys.stderr)
        obs_text = str(obs)
        
    try:
        # Anti-Rate Limit Buffer (Slows down requests slightly to appease Groq)
        time.sleep(1.5) 
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=TEMPERATURE,
            max_tokens=150,  
        )
        raw_output = completion.choices[0].message.content.strip()
        
        match = re.search(r'\[(ALLOW|LABEL_WARNING|REDUCE_REACH|DELETE|ESCALATE)\]', raw_output, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()
            
        #print(f"🚨 REGEX FAILED. LLM Output: {raw_output}", file=sys.stderr)
        return "ESCALATE" 
        
    except Exception as e:
        #print(f"🚨 API ERROR: {e}", file=sys.stderr)
        return "ESCALATE"

async def main() -> None:
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
                #print(f"🚨 ENVIRONMENT/SERVER ERROR: {e}", file=sys.stderr)
                pass
            finally:
                log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())