# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Social Media Moderation Environment Implementation.

This environment simulates a content moderation system where an AI agent
learns to make strategic decisions about social media posts that may or
may not contain misinformation.

The agent never sees ground truth — it only sees noisy signals and must
develop a strategy to contain misinformation while protecting real content.
"""

import random
import uuid
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# try:
#     from ..models import ModerationAction, ModerationObservation
# except ImportError:
#     from models import ModerationAction, ModerationObservation


try:
    # Try absolute import first (works in Docker/Production)
    from models import ModerationAction, ModerationObservation
except ImportError:
    # Fallback to relative import (works for local development)
    from ..models import ModerationAction, ModerationObservation
# ─── Constants ────────────────────────────────────────────────────────────────

CONTENT_CATEGORIES = ["health", "politics", "entertainment", "finance"]

HIGH_RISK_CATEGORIES = ["health", "politics"]

TASK_CONFIGS = {
    1: {
        "total_posts": 10,
        "fake_ratio": 0.40,
        "signal_noise": 0.0,
        "adversarial_report_prob": 0.0,
        "repeat_offender_prob": 0.0,
        "factcheck_rise_rate": 0.08,
        "campaign_size": 0,
    },
    2: {
        "total_posts": 15,
        "fake_ratio": 0.35,
        "signal_noise": 0.15,
        "adversarial_report_prob": 0.2,
        "repeat_offender_prob": 0.2,
        "factcheck_rise_rate": 0.05,
        "campaign_size": 0,
    },
    3: {
        "total_posts": 20,
        "fake_ratio": 0.30,
        "signal_noise": 0.25,
        "adversarial_report_prob": 0.4,
        "repeat_offender_prob": 0.3,
        "factcheck_rise_rate": 0.03,
        "campaign_size": 5,
    },
}

HARM_THRESHOLD = 5.0


# ─── Environment ──────────────────────────────────────────────────────────────

class SocialMediaModerationEnvironment(Environment):
    """
    Social Media Moderation Environment.

    The agent must moderate a stream of social media posts, deciding
    what action to take on each one based on noisy signals.

    Hidden state (agent never sees):
        - is_fake: whether each post is actually misinformation
        - true_spread: actual spread level of each post
        - campaign_post_ids: which posts are part of coordinated campaign

    Observable state (agent sees via ModerationObservation):
        - misinfo_probability, virality_score, spread_velocity
        - report_count, trusted_report_count, reporter_trust
        - user_credibility, is_repeat_offender
        - factcheck_confidence, content_category
        - step_number, posts_remaining
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]

        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._posts: List[Dict] = []
        self._current_post_index: int = 0
        self._cumulative_harm: float = 0.0
        self._consecutive_same_action: int = 0
        self._last_action: Optional[str] = None
        self._delete_count: int = 0
        self._escalated_posts: Dict[int, int] = {}
        self._action_history: List[str] = []
        self._total_posts_processed: int = 0

        # Grader tracking
        self._fake_posts_actioned: int = 0
        self._real_posts_protected: int = 0
        self._total_fake: int = 0
        self._total_real: int = 0
        self._early_actions: int = 0
        self._campaign_posts_actioned: int = 0
        self._total_spread_allowed: float = 0.0

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed: int = 42, task_id: int = None, **kwargs) -> ModerationObservation:
        """Reset the environment and start a new episode."""
        random.seed(seed)

        if task_id is not None:
            self.task_id = task_id
            self.config = TASK_CONFIGS[task_id]

        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_post_index = 0
        self._cumulative_harm = 0.0
        self._consecutive_same_action = 0
        self._last_action = None
        self._delete_count = 0
        self._escalated_posts = {}
        self._action_history = []
        self._total_posts_processed = 0

        self._fake_posts_actioned = 0
        self._real_posts_protected = 0
        self._total_fake = 0
        self._total_real = 0
        self._early_actions = 0
        self._campaign_posts_actioned = 0
        self._total_spread_allowed = 0.0

        self._posts = self._generate_posts()

        for post in self._posts:
            if post["is_fake"]:
                self._total_fake += 1
            else:
                self._total_real += 1

        return self._make_observation(0, reward=0.0, done=False)

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: ModerationAction, **kwargs) -> ModerationObservation:
        """Process agent's moderation action on current post."""
        self._state.step_count += 1
        chosen_action = action.action
        self._action_history.append(chosen_action)

        post = self._posts[self._current_post_index]

        self._resolve_escalations()

        reward = self._calculate_reward(chosen_action, post)
        self._update_tracking(chosen_action, post)
        reward += self._check_degenerate_behaviour(chosen_action)

        if reward < 0:
            self._cumulative_harm += abs(reward)

        self._current_post_index += 1

        total_posts = self.config["total_posts"]
        episode_done = (
            self._current_post_index >= total_posts
            or self._cumulative_harm >= HARM_THRESHOLD
        )

        if episode_done:
            if self._delete_count > total_posts * 0.8:
                reward -= 0.5

        if episode_done:
            obs = self._make_observation(
                self._current_post_index - 1,
                reward=reward,
                done=True
            )
        else:
            obs = self._make_observation(
                self._current_post_index,
                reward=reward,
                done=False
            )

        return obs

    # ─── State ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state

    # ─── Post Generator ───────────────────────────────────────────────────────

    def _generate_posts(self) -> List[Dict]:
        """Generate all posts for this episode."""
        config = self.config
        total = config["total_posts"]
        fake_count = int(total * config["fake_ratio"])
        noise = config["signal_noise"]
        campaign_size = config["campaign_size"]

        posts = []

        fake_indices = set(random.sample(range(total), fake_count))

        campaign_indices = set()
        if campaign_size > 0:
            campaign_indices = set(
                random.sample(list(fake_indices), min(campaign_size, fake_count))
            )

        for i in range(total):
            is_fake = i in fake_indices
            is_campaign = i in campaign_indices
            is_repeat = random.random() < config["repeat_offender_prob"]
            category = random.choice(CONTENT_CATEGORIES)

            if is_fake:
                virality = random.uniform(0.40, 1.0)
            else:
                virality = random.uniform(0.0, 0.70)

            spread_velocity = min(virality * random.uniform(0.5, 1.5), 1.0)

            if is_repeat:
                user_credibility = random.uniform(0.1, 0.4)
            else:
                user_credibility = random.uniform(0.4, 1.0)

            if is_fake:
                base_prob = random.uniform(0.60, 0.95)
            else:
                base_prob = random.uniform(0.05, 0.45)

            misinfo_prob = max(0.0, min(1.0, base_prob + random.uniform(-noise, noise)))

            is_brigaded = (
                not is_fake
                and random.random() < config["adversarial_report_prob"]
            )

            base_reports = int(virality * 100)

            if is_fake:
                report_count = base_reports + random.randint(20, 80)
                trusted_report_count = random.randint(2, 10)
                reporter_trust = random.uniform(0.5, 0.9)
            elif is_brigaded:
                report_count = base_reports + random.randint(50, 150)
                trusted_report_count = random.randint(0, 2)
                reporter_trust = random.uniform(0.1, 0.3)
            else:
                report_count = base_reports + random.randint(0, 30)
                trusted_report_count = random.randint(0, 3)
                reporter_trust = random.uniform(0.3, 0.7)

            report_count = min(report_count, 200)

            posts.append({
                "is_fake": is_fake,
                "is_campaign": is_campaign,
                "is_brigaded": is_brigaded,
                "true_virality": virality,
                "misinfo_probability": round(misinfo_prob, 3),
                "virality_score": round(virality, 3),
                "spread_velocity": round(spread_velocity, 3),
                "report_count": report_count,
                "trusted_report_count": trusted_report_count,
                "reporter_trust": round(reporter_trust, 3),
                "user_credibility": round(user_credibility, 3),
                "is_repeat_offender": is_repeat,
                "content_category": category,
            })

        return posts

    # ─── Reward Calculator ────────────────────────────────────────────────────

    def _calculate_reward(self, action: str, post: Dict) -> float:
        """Calculate reward for agent's action on current post."""
        is_fake = post["is_fake"]
        is_repeat = post["is_repeat_offender"]
        is_viral = post["true_virality"] > 0.6
        category = post["content_category"]
        is_high_risk = category in HIGH_RISK_CATEGORIES
        spread_velocity = post["spread_velocity"]

        reward = 0.0

        if is_fake:
            if action == "DELETE":
                reward = 1.2 if is_repeat else 1.0
            elif action == "LABEL_WARNING":
                reward = -0.5 if is_repeat else 0.6
            elif action == "REDUCE_REACH":
                reward = 0.5
            elif action == "ALLOW":
                if is_viral:
                    reward = -1.5 if is_repeat else -1.0
                    if is_high_risk:
                        reward -= 0.5
                else:
                    reward = -1.5 if is_repeat else -0.5
            elif action == "ESCALATE":
                fc = self._get_current_factcheck(self._current_post_index)
                reward = 0.3 if fc < 0.3 else -0.2
        else:
            if action == "ALLOW":
                reward = 0.4
            elif action == "DELETE":
                reward = -1.5 if is_repeat else -1.2
            elif action == "REDUCE_REACH":
                reward = -0.3
            elif action == "LABEL_WARNING":
                reward = -0.2
            elif action == "ESCALATE":
                fc = self._get_current_factcheck(self._current_post_index)
                reward = 0.3 if fc < 0.3 else -0.2

        if action in ["DELETE", "REDUCE_REACH", "LABEL_WARNING"] and is_fake:
            if spread_velocity > 0.7 and post["true_virality"] < 0.8:
                reward += 0.3

        if action == "ALLOW" and is_fake:
            self._total_spread_allowed += post["true_virality"]

        return round(reward, 3)

    # ─── Degenerate Behaviour Checker ─────────────────────────────────────────

    def _check_degenerate_behaviour(self, action: str) -> float:
        """Penalize agent for lazy/repetitive behaviour."""
        penalty = 0.0

        if action == self._last_action:
            self._consecutive_same_action += 1
            if self._consecutive_same_action >= 5:
                penalty -= 0.2
        else:
            self._consecutive_same_action = 0

        self._last_action = action
        return penalty

    # ─── Escalation Resolver ──────────────────────────────────────────────────

    def _resolve_escalations(self):
        """Resolve pending escalations after 2-3 steps."""
        current_step = self._state.step_count
        resolved = []
        for post_idx, escalated_step in self._escalated_posts.items():
            if current_step - escalated_step >= 2:
                self._posts[post_idx]["factcheck_confidence"] = 1.0
                resolved.append(post_idx)
        for idx in resolved:
            del self._escalated_posts[idx]

    # ─── Tracking ─────────────────────────────────────────────────────────────

    def _update_tracking(self, action: str, post: Dict):
        """Update grader tracking variables."""
        is_fake = post["is_fake"]
        is_strong_action = action in ["DELETE", "REDUCE_REACH", "LABEL_WARNING"]

        self._total_posts_processed += 1

        if is_fake and is_strong_action:
            self._fake_posts_actioned += 1
            if post.get("is_campaign"):
                self._campaign_posts_actioned += 1

        if not is_fake and action == "ALLOW":
            self._real_posts_protected += 1

        if action == "DELETE":
            self._delete_count += 1

        if action == "ESCALATE":
            self._escalated_posts[self._current_post_index] = self._state.step_count

        if is_fake and is_strong_action:
            if post["spread_velocity"] > 0.7 and post["true_virality"] < 0.8:
                self._early_actions += 1

    # ─── Observation Builder ──────────────────────────────────────────────────

    def _make_observation(self, post_index: int, reward: float, done: bool) -> ModerationObservation:
        """Build a ModerationObservation from post data."""
        total_posts = self.config["total_posts"]

        if post_index >= len(self._posts):
            post_index = len(self._posts) - 1

        post = self._posts[post_index]
        step = self._state.step_count

        if post.get("is_campaign"):
            factcheck = 0.1
        else:
            factcheck = min(0.1 + (step * self.config["factcheck_rise_rate"]), 0.95)

        return ModerationObservation(
            misinfo_probability=post["misinfo_probability"],
            virality_score=post["virality_score"],
            spread_velocity=post["spread_velocity"],
            report_count=post["report_count"],
            trusted_report_count=post["trusted_report_count"],
            reporter_trust=post["reporter_trust"],
            user_credibility=post["user_credibility"],
            is_repeat_offender=post["is_repeat_offender"],
            factcheck_confidence=round(factcheck, 3),
            content_category=post["content_category"],
            step_number=step,
            posts_remaining=max(0, total_posts - post_index - 1),
            done=done,
            reward=reward,
        )

    # ─── Helper ───────────────────────────────────────────────────────────────

    def _get_current_factcheck(self, post_index: int) -> float:
        """Get current factcheck confidence for a post."""
        post = self._posts[post_index]
        if post.get("is_campaign"):
            return 0.1
        step = self._state.step_count
        return min(0.1 + (step * self.config["factcheck_rise_rate"]), 0.95)

    # ─── Grader ───────────────────────────────────────────────────────────────

    def get_grader_score(self) -> float:
        """Calculate final grader score. Returns float between 0.0 and 1.0"""
        if self.task_id == 1:
            return self._grade_task1()
        elif self.task_id == 2:
            return self._grade_task2()
        else:
            return self._grade_task3()

    def _grade_task1(self) -> float:
        """
        Task 1 grader — Basic Moderation.
        Score = 60% fake control + 40% real protection
        """
        fake_control = (
            self._fake_posts_actioned / self._total_fake
            if self._total_fake > 0 else 0.0
        )
        real_protection = (
            self._real_posts_protected / self._total_real
            if self._total_real > 0 else 0.0
        )
        over_deletion_penalty = min(
            self._delete_count / max(self._total_posts_processed, 1), 1.0
        )
        penalty = 0.2 if over_deletion_penalty > 0.8 else 0.0

        score = (0.6 * fake_control) + (0.4 * real_protection) - penalty
        return round(min(max(score, 0.0), 1.0), 3)

    def _grade_task2(self) -> float:
        """
        Task 2 grader — Balancing Act.
        Score = 50% fake control + 35% real protection + 15% early action
        """
        fake_control = (
            self._fake_posts_actioned / self._total_fake
            if self._total_fake > 0 else 0.0
        )
        real_protection = (
            self._real_posts_protected / self._total_real
            if self._total_real > 0 else 0.0
        )
        early_action_rate = (
            self._early_actions / self._total_fake
            if self._total_fake > 0 else 0.0
        )
        balance_penalty = 0.2 if (fake_control > 0 and real_protection < 0.3) else 0.0

        score = (
            (0.50 * fake_control)
            + (0.35 * real_protection)
            + (0.15 * early_action_rate)
            - balance_penalty
        )
        return round(min(max(score, 0.0), 1.0), 3)

    def _grade_task3(self) -> float:
        """
        Task 3 grader — Campaign Detection.
        Score = 40% campaign detection + 35% non-campaign accuracy + 25% harm reduction
        """
        campaign_detection = (
            self._campaign_posts_actioned / self.config["campaign_size"]
            if self.config["campaign_size"] > 0 else 0.0
        )
        non_campaign_fake = self._total_fake - self.config["campaign_size"]
        non_campaign_actioned = self._fake_posts_actioned - self._campaign_posts_actioned
        non_campaign_accuracy = (
            non_campaign_actioned / non_campaign_fake
            if non_campaign_fake > 0 else 0.0
        )
        max_possible_spread = sum(p["true_virality"] for p in self._posts if p["is_fake"])
        harm_reduction = (
            1.0 - (self._total_spread_allowed / max_possible_spread)
            if max_possible_spread > 0 else 1.0
        )
        real_protection = (
            self._real_posts_protected / self._total_real
            if self._total_real > 0 else 0.0
        )
        if real_protection < 0.4:
            protection_penalty = 0.3
        elif real_protection < 0.6:
            protection_penalty = 0.15
        else:
            protection_penalty = 0.0

        score = (
            (0.40 * campaign_detection)
            + (0.35 * non_campaign_accuracy)
            + (0.25 * harm_reduction)
            - protection_penalty
        )
        return round(min(max(score, 0.0), 1.0), 3)