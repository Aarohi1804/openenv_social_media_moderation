# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Social Media Moderation Environment.

This environment simulates a content moderation systen where an AI agent learns
to make strategic decisions about social media posts that may or may not 
contain misinformation.
"""

from typing import Literal
from openenv.core.env_server.types import Action,Observation
from pydantic import Field

class ModerationAction(Action):
    """
    Action taken by the agent on a social media post.

    The agent must choose one of the 5 actions:
    -ALLOW: Leave the post as it is.
    -LABEL_WARNING: add a warning label to the post
    -REDUCE_REACH: Limit how many people see the post.  
    -DELETE: Remove the post completely.
    -ESCALATE: Send to human reveiw.
    """
    action: Literal[
        "ALLOW",
        "LABEL_WARNING",
        "REDUCE_REACH",
        "DELETE",
        "ESCALATE"
    ] =Field(..., description= "Moderation action to be taken on the post")
    
class ModerationObservation(Observation):
    """
    What the agent sees about each incoming social media post.
    
    The agent never sees the ground truth (is_fake).
    It only sees these noisy signals and must decide strategically.
    """
    
    # Core misinformation signals
    misinfo_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated probability that this post contains misinformation (0.0-1.0)"
    )
    virality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How fast this post is currently spreading (0.0-1.0)"
    )
    spread_velocity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rate at which virality is growing; how fast it is accelerating (0.0-1.0)"
    )

    # Reporting signals
    report_count: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Total number of users who reported this post — can be inflated by brigading (0.0-1.0)"
    )
    trusted_report_count: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Number of high-credibility reporters(trust >0.7) who flagged this post."
    )
    reporter_trust: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trust of high-credibility of the users who reported this post (0.0-1.0)"
    )

    # User signals
    user_credibility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trustworthiness of the user who posted this content (0.0-1.0)"
    )
    is_repeat_offender: bool = Field(
        default=False,
        description="Whether this user has had posts actioned before."
    )

    # Fact checking signals
    factcheck_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="How much fact-checkers have reviewed this post — starts low, rises over time (0.0-1.0)"
    )

    # Content signal
    content_category: Literal[
        "health",
        "politics",
        "entertainment",
        "finance"
    ] = Field(
        default="entertainment",
        description="Topic category of the post"
    )
    # Episode progress info
    step_number: int = Field(
        default=0,
        description="The current step number in the episode"
    )
    posts_remaining: int = Field(
        default=0,
        description="Number of posts remaining in this episode"
    )

        