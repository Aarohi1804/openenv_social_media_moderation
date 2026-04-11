# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

# ─── INDESTRUCTIBLE IMPORT FIX ──────────────────────────────────────────────
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

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

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Phase 2 Validation Ready."}

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

# ─── THE SENIOR'S EXACT FIX ──────────────────────────────────────────────────
@app.get("/grader")
def get_grader():
    try:
        # Grabs the running instance WITHOUT needing the session_id
        env_instance = list(app.state.envs.values())[0]
        raw_score = env_instance.get_grader_score()
    except Exception:
        raw_score = 0.01

    # Exact clamp from April 7th
    return {"score": float(max(0.01, min(0.99, raw_score)))}
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()