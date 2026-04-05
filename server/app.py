# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Social Media Moderation Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    # Try absolute import first (works in Docker/Production)
    from models import ModerationAction, ModerationObservation
    from server.social_media_moderation_env_environment import SocialMediaModerationEnvironment
except (ImportError, ModuleNotFoundError):
    # Fallback to relative import (works for local development)
    from ..models import ModerationAction, ModerationObservation
    from .social_media_moderation_env_environment import SocialMediaModerationEnvironment

# Create the FastAPI app
# This one line creates ALL endpoints: /ws, /reset, /step, /state, /health, /docs
app = create_app(
    SocialMediaModerationEnvironment,
    ModerationAction,
    ModerationObservation,
    env_name="social_media_moderation_env",
    max_concurrent_envs=100,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()
#     main(port=args.port)
if __name__ == '__main__':
    main()

    