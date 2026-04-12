

"""Social Media Moderation Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Importing the models and environment client for the Social Media Moderation Environment
from models import ModerationAction, ModerationObservation

class SocialMediaModerationEnv(
    EnvClient[ModerationAction, ModerationObservation, State]
):
    """
    Client for the Social Media Moderation Environment.
    """

    def _step_payload(self, action: ModerationAction) -> Dict:
        """Convert ModerationAction to JSON payload for step message."""
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ModerationObservation]:
        """Parse server response into StepResult[ModerationObservation]."""
        obs_data = payload.get("observation", {})

        observation = ModerationObservation(
            misinfo_probability=obs_data.get("misinfo_probability", 0.0),
            virality_score=obs_data.get("virality_score", 0.0),
            spread_velocity=obs_data.get("spread_velocity", 0.0),
            report_count=obs_data.get("report_count", 0),
            trusted_report_count=obs_data.get("trusted_report_count", 0),
            reporter_trust=obs_data.get("reporter_trust", 0.5),
            user_credibility=obs_data.get("user_credibility", 0.5),
            is_repeat_offender=obs_data.get("is_repeat_offender", False),
            factcheck_confidence=obs_data.get("factcheck_confidence", 0.1),
            content_category=obs_data.get("content_category", "entertainment"),
            step_number=obs_data.get("step_number", 0),
            posts_remaining=obs_data.get("posts_remaining", 0),
            environmental_warning=obs_data.get("environmental_warning", "None"),
            user_id=obs_data.get("user_id", "unknown"),
            dynamic_reputation=obs_data.get("dynamic_reputation", 0.7),
            
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
    