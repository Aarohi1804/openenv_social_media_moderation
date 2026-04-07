# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import uuid
import sys
import os
from typing import Dict, List, Optional

# ─── INDESTRUCTIBLE IMPORT FIX ──────────────────────────────────────────────
# This ensures the server can find 'models.py' no matter how it starts the app.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    # 1. Try the Scaler server's absolute path
    from server.models import ModerationAction, ModerationObservation
except ImportError:
    try:
        # 2. Try the local root path
        from models import ModerationAction, ModerationObservation
    except ImportError:
        # 3. Final fallback
        from .models import ModerationAction, ModerationObservation

# ─── Constants ────────────────────────────────────────────────────────────────

CONTENT_CATEGORIES = ["health", "politics", "entertainment", "finance"]
HIGH_RISK_CATEGORIES = ["health", "politics"]

TASK_CONFIGS = {
    "task_easy": {
        "total_posts": 10,
        "fake_ratio": 0.40,
        "signal_noise": 0.0,
        "adversarial_report_prob": 0.0,
        "repeat_offender_prob": 0.0,
        "factcheck_rise_rate": 0.08,
        "campaign_size": 0,
    },
    "task_medium": {
        "total_posts": 15,
        "fake_ratio": 0.35,
        "signal_noise": 0.15,
        "adversarial_report_prob": 0.2,
        "repeat_offender_prob": 0.2,
        "factcheck_rise_rate": 0.05,
        "campaign_size": 0,
    },
    "task_hard": {
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
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def _normalize_task_id(self, task_id) -> str:
        tid = str(task_id).lower().strip()
        if tid in ["task_easy", "task1", "1", "easy"]:
            return "task_easy"
        elif tid in ["task_medium", "task2", "2", "medium"]:
            return "task_medium"
        elif tid in ["task_hard", "task3", "3", "hard"]:
            return "task_hard"
        return "task_easy"

    def __init__(self, task_id="task_easy"):
        self.task_id = self._normalize_task_id(task_id)
        self.config = TASK_CONFIGS[self.task_id]

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

        self._fake_posts_actioned: int = 0
        self._real_posts_protected: int = 0
        self._total_fake: int = 0
        self._total_real: int = 0
        self._early_actions: int = 0
        self._campaign_posts_actioned: int = 0
        self._total_spread_allowed: float = 0.0



    def reset(self, seed: int = 42, task_id=None, **kwargs) -> ModerationObservation:

        random.seed(seed)

        if task_id is not None:
            self.task_id = self._normalize_task_id(task_id)
            self.config = TASK_CONFIGS[self.task_id]

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
            if post["is_fake"]: self._total_fake += 1
            else: self._total_real += 1
        return self._make_observation(0, reward=0.0, done=False)


    def step(self, action: ModerationAction, **kwargs) -> ModerationObservation:

        self._state.step_count += 1
        chosen_action = action.action
        self._action_history.append(chosen_action)


        post = self._posts[self._current_post_index]


        self._resolve_escalations()

        
        raw_reward = self._calculate_reward(chosen_action, post)
        self._update_tracking(chosen_action, post)
        raw_reward += self._check_degenerate_behaviour(chosen_action)
        if raw_reward < 0: self._cumulative_harm += abs(raw_reward)
        self._current_post_index += 1
        total_posts = self.config["total_posts"]
        episode_done = (self._current_post_index >= total_posts or self._cumulative_harm >= HARM_THRESHOLD)
        if episode_done and self._delete_count > total_posts * 0.8: raw_reward -= 0.5
        normalized_reward = round(max(0.01, min(0.99, (raw_reward + 2.5) / 5.0)), 3)
        return self._make_observation(self._current_post_index if not episode_done else self._current_post_index - 1, reward=normalized_reward, done=episode_done)

    @property
    def state(self) -> State: return self._state

    def _generate_posts(self) -> List[Dict]:

        config = self.config
        total = config["total_posts"]
        fake_indices = set(random.sample(range(total), int(total * config["fake_ratio"])))
        campaign_indices = set(random.sample(list(fake_indices), min(config["campaign_size"], len(fake_indices)))) if config["campaign_size"] > 0 else set()
        posts = []
        for i in range(total):
            is_fake = i in fake_indices
            is_campaign = i in campaign_indices
            is_repeat = random.random() < config["repeat_offender_prob"]
            category = random.choice(CONTENT_CATEGORIES)
            virality = random.uniform(0.4, 1.0) if is_fake else random.uniform(0.0, 0.7)
            spread_vel = min(virality * random.uniform(0.5, 1.5), 1.0)
            user_cred = random.uniform(0.1, 0.4) if is_repeat else random.uniform(0.4, 1.0)
            base_prob = random.uniform(0.6, 0.95) if is_fake else random.uniform(0.05, 0.45)
            misinfo_prob = max(0.0, min(1.0, base_prob + random.uniform(-config["signal_noise"], config["signal_noise"])))
            is_brigaded = not is_fake and random.random() < config["adversarial_report_prob"]
            base_reps = int(virality * 100)
            if is_fake:
                reps, treps, trust = base_reps + random.randint(20, 80), random.randint(2, 10), random.uniform(0.5, 0.9)
            elif is_brigaded:
                reps, treps, trust = base_reps + random.randint(50, 150), random.randint(0, 2), random.uniform(0.1, 0.3)
            else:
                reps, treps, trust = base_reps + random.randint(0, 30), random.randint(0, 3), random.uniform(0.3, 0.7)
            posts.append({
                "is_fake": is_fake, "is_campaign": is_campaign, "is_brigaded": is_brigaded,
                "true_virality": virality, "misinfo_probability": round(misinfo_prob, 3),
                "virality_score": round(virality, 3), "spread_velocity": round(spread_vel, 3),
                "report_count": min(reps, 200), "trusted_report_count": treps,
                "reporter_trust": round(trust, 3), "user_credibility": round(user_cred, 3),
                "is_repeat_offender": is_repeat, "content_category": category,
            })
        return posts

    def _calculate_reward(self, action: str, post: Dict) -> float:
        is_fake, is_repeat = post["is_fake"], post["is_repeat_offender"]
        is_viral, is_high_risk = post["true_virality"] > 0.6, post["content_category"] in HIGH_RISK_CATEGORIES
        reward = 0.0
        if is_fake:
            if action == "DELETE": reward = 1.2 if is_repeat else 1.0
            elif action == "LABEL_WARNING": reward = -0.5 if is_repeat else 0.6
            elif action == "REDUCE_REACH": reward = 0.5
            elif action == "ALLOW": reward = -1.5 if is_repeat else -1.0; reward -= 0.5 if is_viral and is_high_risk else 0
            elif action == "ESCALATE": reward = 0.3 if self._get_current_factcheck(self._current_post_index) < 0.3 else -0.2
        else:
            if action == "ALLOW": reward = 0.4
            elif action == "DELETE": reward = -1.5 if is_repeat else -1.2
            elif action == "REDUCE_REACH": reward = -0.3
            elif action == "LABEL_WARNING": reward = -0.2
            elif action == "ESCALATE": reward = 0.3 if self._get_current_factcheck(self._current_post_index) < 0.3 else -0.2
        if action in ["DELETE", "REDUCE_REACH", "LABEL_WARNING"] and is_fake and post["spread_velocity"] > 0.7: reward += 0.3
        if action == "ALLOW" and is_fake: self._total_spread_allowed += post["true_virality"]
        return round(reward, 3)

    def _check_degenerate_behaviour(self, action: str) -> float:
        penalty = -0.2 if action == self._last_action and self._consecutive_same_action >= 5 else 0.0
        self._consecutive_same_action = self._consecutive_same_action + 1 if action == self._last_action else 0
        self._last_action = action
        return penalty

    def _resolve_escalations(self):
        current_step, resolved = self._state.step_count, []
        for p_idx, e_step in self._escalated_posts.items():
            if current_step - e_step >= 2: self._posts[p_idx]["factcheck_confidence"] = 1.0; resolved.append(p_idx)
        for idx in resolved: del self._escalated_posts[idx]

    def _update_tracking(self, action: str, post: Dict):
        is_fake, is_strong = post["is_fake"], action in ["DELETE", "REDUCE_REACH", "LABEL_WARNING"]
        self._total_posts_processed += 1
        if is_fake and is_strong:
            self._fake_posts_actioned += 1
            if post.get("is_campaign"): self._campaign_posts_actioned += 1
            if post["spread_velocity"] > 0.7: self._early_actions += 1
        if not is_fake and action == "ALLOW": self._real_posts_protected += 1
        if action == "DELETE": self._delete_count += 1
        if action == "ESCALATE": self._escalated_posts[self._current_post_index] = self._state.step_count

    def _make_observation(self, post_index: int, reward: float, done: bool) -> ModerationObservation:
        p_idx = min(post_index, len(self._posts) - 1)
        post, step = self._posts[p_idx], self._state.step_count
        fc = 0.1 if post.get("is_campaign") else min(0.1 + (step * self.config["factcheck_rise_rate"]), 0.95)
        return ModerationObservation(
            misinfo_probability=post["misinfo_probability"], virality_score=post["virality_score"],
            spread_velocity=post["spread_velocity"], report_count=post["report_count"],
            trusted_report_count=post["trusted_report_count"], reporter_trust=post["reporter_trust"],
            user_credibility=post["user_credibility"], is_repeat_offender=post["is_repeat_offender"],
            factcheck_confidence=round(fc, 3), content_category=post["content_category"],
            step_number=step, posts_remaining=max(0, self.config["total_posts"] - p_idx - 1),
            done=done, reward=reward,
        )

    def _get_current_factcheck(self, post_index: int) -> float:
        post = self._posts[post_index]
        return 0.1 if post.get("is_campaign") else min(0.1 + (self._state.step_count * self.config["factcheck_rise_rate"]), 0.95)

    def get_grader_score(self) -> float:
        if self.task_id == "task_easy": final_score = self._grade_task1()
        elif self.task_id == "task_medium": final_score = self._grade_task2()
        elif self.task_id == "task_hard": final_score = self._grade_task3()
        else: final_score = 0.5
        return float(max(0.01, min(0.99, final_score)))

    def _grade_task1(self) -> float:
        f_ctrl = self._fake_posts_actioned / self._total_fake if self._total_fake > 0 else 0.0
        r_prot = self._real_posts_protected / self._total_real if self._total_real > 0 else 0.0
        penalty = 0.2 if (self._delete_count / max(self._total_posts_processed, 1)) > 0.8 else 0.0
        return round(max(0.001, min(0.999, (0.6 * f_ctrl) + (0.4 * r_prot) - penalty)), 3)

    def _grade_task2(self) -> float:
        f_ctrl = self._fake_posts_actioned / self._total_fake if self._total_fake > 0 else 0.0
        r_prot = self._real_posts_protected / self._total_real if self._total_real > 0 else 0.0
        early = self._early_actions / self._total_fake if self._total_fake > 0 else 0.0
        bal = 0.2 if (f_ctrl > 0 and r_prot < 0.3) else 0.0
        return round(max(0.001, min(0.999, (0.5 * f_ctrl) + (0.35 * r_prot) + (0.15 * early) - bal)), 3)

    def _grade_task3(self) -> float:
        camp = self._campaign_posts_actioned / self.config["campaign_size"] if self.config["campaign_size"] > 0 else 0.0
        n_camp_f = self._total_fake - self.config["campaign_size"]
        n_camp_a = self._fake_posts_actioned - self._campaign_posts_actioned
        n_camp_acc = n_camp_a / n_camp_f if n_camp_f > 0 else 0.0
        max_spread = sum(p["true_virality"] for p in self._posts if p["is_fake"])
        h_red = (1.0 - (self._total_spread_allowed / max_spread) if max_spread > 0 else 1.0)
        r_prot = self._real_posts_protected / self._total_real if self._total_real > 0 else 0.0
        p_pen = 0.3 if r_prot < 0.4 else (0.15 if r_prot < 0.6 else 0.0)
        return round(max(0.001, min(0.999, (0.4 * camp) + (0.35 * n_camp_acc) + (0.25 * h_red) - p_pen)), 3)


# ─── EXTERNAL GRADERS (STRICT FORMAT WITH SAFETY NET) ─────────────────────

# ─── EXTERNAL GRADERS (SIR'S EDGE-CASE FIX) ───────────────────────────────

class EasyGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            real_env = getattr(env, "unwrapped", env)
            raw_score = float(real_env._grade_task1())
            return max(0.01, min(0.99, raw_score))
        except Exception:
            # Handles Sir's "empty input" edge cases safely
            return 0.01

class MediumGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            real_env = getattr(env, "unwrapped", env)
            raw_score = float(real_env._grade_task2())
            return max(0.01, min(0.99, raw_score))
        except Exception:
            return 0.01

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            real_env = getattr(env, "unwrapped", env)
            raw_score = float(real_env._grade_task3())
            return max(0.01, min(0.99, raw_score))
        except Exception:
            return 0.01