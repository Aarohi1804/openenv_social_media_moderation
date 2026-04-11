# Copyright (c) Meta Platforms, Inc. and affiliates.
import random
import uuid
import sys
import os
import traceback
from typing import Dict, List, Optional, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models import ModerationAction, ModerationObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# --- SENIOR'S EPSILON CLAMPING (PHASE 2 FIX) ---
def clamp_score(score: float) -> float:
    """Ensures score is strictly between 0 and 1 (e.g., 0.001 to 0.999)."""
    epsilon = 0.001
    try:
        val = float(score)
    except (ValueError, TypeError):
        return epsilon
    return max(epsilon, min(1.0 - epsilon, val))

CONTENT_CATEGORIES = ["health", "politics", "entertainment", "finance"]
HIGH_RISK_CATEGORIES = ["health", "politics"]

TASK_CONFIGS = {
    "task_easy": { "total_posts": 10, "fake_ratio": 0.40, "signal_noise": 0.0, "adversarial_report_prob": 0.0, "repeat_offender_prob": 0.0, "factcheck_rise_rate": 0.08, "campaign_size": 0 },
    "task_medium": { "total_posts": 15, "fake_ratio": 0.35, "signal_noise": 0.15, "adversarial_report_prob": 0.2, "repeat_offender_prob": 0.2, "factcheck_rise_rate": 0.05, "campaign_size": 0 },
    "task_hard": { "total_posts": 20, "fake_ratio": 0.30, "signal_noise": 0.25, "adversarial_report_prob": 0.4, "repeat_offender_prob": 0.3, "factcheck_rise_rate": 0.03, "campaign_size": 5 }
}

HARM_THRESHOLD = 5.0

class SocialMediaModerationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    @property
    def state(self) -> State:
        return self._state

    def _normalize_task_id(self, task_id: str) -> str:
        tid = str(task_id).lower().strip()
        if tid in ["task_easy", "task1", "1", "easy"]: return "task_easy"
        elif tid in ["task_medium", "task2", "2", "medium"]: return "task_medium"
        elif tid in ["task_hard", "task3", "3", "hard"]: return "task_hard"
        return "task_easy"

    def __init__(self, task_id: str = "task_easy"):
        try:
            self.task_id = self._normalize_task_id(task_id)
            self.config = TASK_CONFIGS[self.task_id]
            self.reset()
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e

    def reset(self, seed: int = 42, task_id: Optional[str] = None) -> ModerationObservation:
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
        self.user_registry = {}
        
        self._posts = self._generate_posts()
        for post in self._posts:
            if post["is_fake"]: self._total_fake += 1
            else: self._total_real += 1
            
        return self._make_observation(0, reward=0.0, done=False)

    def step(self, action: ModerationAction) -> ModerationObservation:
        self._state.step_count += 1
        chosen_action = action.action
        
        # Track consecutive actions for degeneracy penalty
        if chosen_action == self._last_action:
            self._consecutive_same_action += 1
        else:
            self._consecutive_same_action = 1
        self._last_action = chosen_action

        post = self._posts[self._current_post_index]
        self._resolve_escalations()

        raw_reward = self._calculate_reward(chosen_action, post)
        self._update_tracking(chosen_action, post)
        
        # Apply behavior penalty
        if self._consecutive_same_action >= 5:
            raw_reward -= 0.2
            
        if raw_reward < 0:
            self._cumulative_harm += abs(raw_reward)
            
        self._current_post_index += 1
        total_posts = self.config["total_posts"]
        episode_done = (self._current_post_index >= total_posts or self._cumulative_harm >= HARM_THRESHOLD)
        
        # Over-moderation penalty
        if episode_done and self._delete_count > total_posts * 0.8:
            raw_reward -= 0.5
            
        normalized_reward = round(max(0.01, min(0.99, (raw_reward + 2.5) / 5.0)), 3)
        return self._make_observation(
            self._current_post_index if not episode_done else self._current_post_index - 1, 
            reward=normalized_reward, 
            done=episode_done
        )

    def _generate_posts(self) -> List[Dict]:
        config = self.config
        total = config["total_posts"]
        user_pool = [f"user_{i}" for i in range(1, 21)]
        fake_indices = set(random.sample(range(total), int(total * config["fake_ratio"])))
        campaign_indices = set(random.sample(list(fake_indices), min(config["campaign_size"], len(fake_indices)))) if config["campaign_size"] > 0 else set()
        
        posts = []
        for i in range(total):
            current_user_id = random.choice(user_pool)
            is_fake = i in fake_indices
            is_campaign = i in campaign_indices
            is_repeat = random.random() < config["repeat_offender_prob"]
            category = random.choice(CONTENT_CATEGORIES)
            
            if is_campaign:
                virality, spread_vel, user_cred, misinfo_prob = 0.99, 0.99, 0.02, 0.98
                reps, treps, trust, is_brigaded = 850, 45, 0.95, True
            else:
                virality = random.uniform(0.4, 1.0) if is_fake else random.uniform(0.0, 0.7)
                spread_vel = min(virality * random.uniform(0.5, 1.5), 1.0)
                user_cred = random.uniform(0.1, 0.4) if is_repeat else random.uniform(0.4, 1.0)
                base_prob = random.uniform(0.6, 0.95) if is_fake else random.uniform(0.05, 0.45)
                misinfo_prob = max(0.0, min(1.0, base_prob + random.uniform(-config["signal_noise"], config["signal_noise"])))
                is_brigaded = not is_fake and random.random() < config["adversarial_report_prob"]
                base_reps = int(virality * 100)
                
                if is_fake: reps, treps, trust = base_reps + random.randint(20, 80), random.randint(2, 10), random.uniform(0.5, 0.9)
                elif is_brigaded: reps, treps, trust = base_reps + random.randint(50, 150), random.randint(0, 2), random.uniform(0.1, 0.3)
                else: reps, treps, trust = base_reps + random.randint(0, 30), random.randint(0, 3), random.uniform(0.3, 0.7)
            
            posts.append({
                "user_id": current_user_id, "is_fake": is_fake, "is_campaign": is_campaign, "is_brigaded": is_brigaded,
                "true_virality": virality, "misinfo_probability": round(misinfo_prob, 3),
                "virality_score": round(virality, 3), "spread_velocity": round(spread_vel, 3),
                "report_count": reps, "trusted_report_count": treps,
                "reporter_trust": round(trust, 3), "user_credibility": round(user_cred, 3),
                "is_repeat_offender": is_repeat, "content_category": category,
            })
        return posts

    def _calculate_reward(self, action: str, post: Dict) -> float:
        is_fake, is_repeat, virality = post["is_fake"], post["is_repeat_offender"], post["true_virality"]
        is_high_risk = post["content_category"] in HIGH_RISK_CATEGORIES
        reward = 0.0
        
        if is_fake:
            if action == "DELETE": reward = (1.2 if is_repeat else 1.0) + (virality * 0.5)
            elif action == "REDUCE_REACH": reward = 0.5 + (virality * 0.2)
            elif action == "LABEL_WARNING": reward = -0.5 if is_repeat else 0.6
            elif action == "ALLOW": reward = (-1.5 if is_repeat else -1.0) * (1.0 + (virality * 2.0)) * (1.5 if is_high_risk else 1.0)
            elif action == "ESCALATE": reward = 0.3 if self._get_current_factcheck(self._current_post_index) < 0.3 else -0.2
        else:
            if action == "ALLOW": reward = 0.4 + (virality * 0.3)
            elif action == "DELETE": reward = (-1.5 if is_repeat else -1.2) * (1.0 + (virality * 2.5))
            elif action == "REDUCE_REACH": reward = -0.3 * (1.0 + virality)
            elif action == "LABEL_WARNING": reward = -0.2 * (1.0 + virality)
            elif action == "ESCALATE": reward = 0.3 if self._get_current_factcheck(self._current_post_index) < 0.3 else -0.2
                
        if action in ["DELETE", "REDUCE_REACH"] and is_fake and post["spread_velocity"] > 0.7: reward += 0.4
        return round(reward, 3)

    def _update_tracking(self, action: str, post: Dict) -> None:
        self._total_posts_processed += 1
        is_fake = post["is_fake"]
        if action == "DELETE": self._delete_count += 1
        if action in ["DELETE", "REDUCE_REACH"] and is_fake:
            self._fake_posts_actioned += 1
            if post.get("is_campaign"): self._campaign_posts_actioned += 1
            if post["spread_velocity"] < 0.6: self._early_actions += 1
        elif action == "ALLOW" and not is_fake:
            self._real_posts_protected += 1
        elif action == "ESCALATE":
            self._escalated_posts[self._current_post_index] = self._state.step_count

    def _resolve_escalations(self):
        # Simulation of human review resolving after 2 steps
        to_resolve = []
        for idx, step_started in self._escalated_posts.items():
            if self._state.step_count - step_started >= 2:
                to_resolve.append(idx)
        for idx in to_resolve:
            del self._escalated_posts[idx]

    def _get_current_factcheck(self, index: int) -> float:
        if index in self._escalated_posts: return 1.0
        rise = self.config["factcheck_rise_rate"] * self._state.step_count
        return min(0.95, 0.1 + rise)

    def _make_observation(self, index: int, reward: float, done: bool) -> ModerationObservation:
        post = self._posts[index]
        return ModerationObservation(
            misinfo_probability=post["misinfo_probability"],
            virality_score=post["virality_score"],
            spread_velocity=post["spread_velocity"],
            report_count=post["report_count"],
            trusted_report_count=post["trusted_report_count"],
            reporter_trust=post["reporter_trust"],
            user_credibility=post["user_credibility"],
            is_repeat_offender=post["is_repeat_offender"],
            factcheck_confidence=self._get_current_factcheck(index),
            environmental_warning=0.8 if post["is_brigaded"] else 0.1,
            content_category=post["content_category"],
            reward=reward,
            done=done
        )

    def get_grader_score(self) -> float:
        """CALLED BY SCALER PHASE 2 VALIDATOR"""
        fake_rate = self._fake_posts_actioned / max(1, self._total_fake)
        real_rate = self._real_posts_protected / max(1, self._total_real)
        
        if self.task_id == "task_easy":
            score = (0.6 * fake_rate) + (0.4 * real_rate)
        elif self.task_id == "task_medium":
            early_rate = self._early_actions / max(1, self._total_fake)
            balance_penalty = abs(fake_rate - real_rate) * 0.2
            score = (0.5 * fake_rate) + (0.35 * real_rate) + (0.15 * early_rate) - balance_penalty
        else: # task_hard
            camp_rate = self._campaign_posts_actioned / max(1, self.config["campaign_size"])
            score = (0.4 * camp_rate) + (0.35 * fake_rate) + (0.25 * real_rate)
            if real_rate < 0.4: score -= 0.3 # Heavy penalty for censorship
            
        return clamp_score(score)