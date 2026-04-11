# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import os

# ─── INDESTRUCTIBLE IMPORT FIX ──────────────────────────────────────────────
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install dependencies with 'uv sync'") from e

from models import ModerationAction, ModerationObservation
from server.social_media_moderation_env_environment import SocialMediaModerationEnvironment

# Create the FastAPI app
app = create_app(
    SocialMediaModerationEnvironment,
    ModerationAction,
    ModerationObservation,
    env_name="social_media_moderation_env",
    max_concurrent_envs=100,
)

# ───  FIX: EXPLICIT ENDPOINTS ─────────────────────────
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "The environment is running perfectly."}

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Social Media: Content Moderation (Easy)",
                "description": "Identify and remove fake posts while protecting real content.",
                "difficulty": "easy"
            },
            {
                "id": "task_medium",
                "name": "Social Media: Advanced Filtering (Medium)",
                "description": "Handle complex moderation scenarios with a focus on early detection.",
                "difficulty": "medium"
            },
            {
                "id": "task_hard",
                "name": "Social Media: Campaign Suppression (Hard)",
                "description": "Mitigate viral fake news campaigns and reduce harmful spread.",
                "difficulty": "hard"
            }
        ],
        "action_schema": ModerationAction.model_json_schema()
    }

@app.get("/grader/{session_id}")
def get_grader(session_id: str):
    try:
        # Fetch the exact environment instance using the session ID
        env_instance = app.state.envs.get(session_id)
        if not env_instance:
            return {"score": 0.01, "error": "Session not found or already closed."}
            
        raw_score = env_instance.get_grader_score()
    except Exception as e:
        print(f"Grader error: {e}")
        raw_score = 0.01

    return {"score": float(max(0.01, min(0.99, raw_score)))}
# ──────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    # This function is what the validator is hunting for
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()