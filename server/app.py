# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

from models import ModerationAction, ModerationObservation
from server.social_media_moderation_env_environment import SocialMediaModerationEnvironment

app = create_app(
    SocialMediaModerationEnvironment,
    ModerationAction,
    ModerationObservation,
    env_name="social_media_moderation_env",
    max_concurrent_envs=100,
)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Phase 2 Validation Ready (String-Type Fix Applied)."}

@app.get("/grader/{session_id}")
def get_grader(session_id: str):
    try:
        env_instance = app.state.envs.get(session_id)
        if not env_instance: return {"score": 0.001}
        raw_score = env_instance.get_grader_score()
    except Exception:
        raw_score = 0.001
    return {"score": float(max(0.001, min(0.999, raw_score)))}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()