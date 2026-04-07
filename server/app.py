# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Social Media Moderation Environment.
"""

import sys
import os

# ─── INDESTRUCTIBLE IMPORT FIX ──────────────────────────────────────────────
# Point Python to the root directory to find models.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install dependencies with 'uv sync'") from e

# Now these imports will work perfectly on the Scaler server
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

def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()