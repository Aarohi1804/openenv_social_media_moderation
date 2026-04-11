# Copyright (c) Meta Platforms, Inc. and affiliates.
import random
import uuid
import sys
import os
import traceback
from typing import Dict, List, Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models import ModerationAction, ModerationObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

    # 🚨 SCRUBBED: Zero kwargs.
    def __init__(self, task_id: str = "task_easy"): 
        try:
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
            self.user_registry = {} 
        except Exception as e:
            print("🚨 CRASH IN __init__ 🚨", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e

    # 🚨 SCRUBBED: Zero kwargs.
    def reset(self, seed: int = 42, task_id: Optional[str] = None) -> ModerationObservation:
        try:
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
        except Exception as e:
            print("🚨 CRASH IN reset 🚨", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e

    # 🚨 SCRUBBED: Zero kwargs.
    def step(self, action: ModerationAction) -> ModerationObservation:
        self._state.step_count += 1
        chosen_action = action.action
        self._action_history.append(chosen_action)
        post = self._posts[self._current_post_index]
        self._resolve_escalations()

        uid = post.get("user_id", "unknown")
        is_fake = post["is_fake"]
        
        if uid in self.user_registry:
            if is_fake and chosen_action == "ALLOW":
                self.user_registry[uid] = max(0.0, self.user_registry[uid] - 0.30)
            elif is_fake and chosen_action in ["DELETE", "REDUCE_REACH", "LABEL_WARNING"]:
                self.user_registry[uid] = max(0.0, self.user_registry[uid] - 0.15)
            elif not is_fake and chosen_action == "ALLOW":
                self.user_registry[uid] = min(1.0, self.user_registry[uid] + 0.05)

        raw_reward = self._calculate_reward(chosen_action, post)
        self._update_tracking(chosen_action, post)
        raw_reward += self._check_degenerate_behaviour(chosen_action)
        
        if raw_reward < 0: 
            self._cumulative_harm += abs(raw_reward)
            
        self._current_post_index += 1
        total_posts = self.config["total_posts"]
        episode_done = (self._current_post_index >= total_posts or self._cumulative_harm >= HARM_THRESHOLD)
        
        if episode_done and self._delete_count > total_posts * 0.8: 
            raw_reward -= 0.5
            
        normalized_reward = round(max(0.01, min(0.99, (raw_reward + 2.5) / 5.0)), 3)
        return self._make_observation(self._current_post_index if not episode_done else self._current_post_index - 1, reward=normalized_reward, done=episode_done)

    def _generate_posts(self) -> List[Dict]:
        config = self.config
        total = config["total_posts"]
        user_pool = [f"user_{i}" for i in range(1, 21)]
        fake_indices = set(random.sample(range(total), int(total * config["fake_ratio"])))
        campaign_indices = set(random.sample(list(fake_indices), min(config["campaign_size"], len(fake_indices)))) if config["campaign_size"] > 0 else set()
        
        posts = []
        for i in range(total):
            current_user_id = random.choice(user_pool)
            if current_user_id not in self.user_registry:
                self.user_registry[current_user_id] = 0.7 

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
        if action == "ALLOW" and is_fake: self._total_spread_allowed += virality
        return round(reward, 3)
    def _update_tracking(self, action: str, post: Dict) -> None:
        self._total_posts_processed += 1
        is_fake = post["is_fake"]

        if action == "DELETE":
            self._delete_count += 1

        # Track successful takedowns of misinformation
        if action in ["DELETE", "REDUCE_REACH"]:
            if is_fake:
                self._fake_posts_actioned += 1
                if post.get("is_campaign"):
                    self._campaign_posts_actioned += 1
                if post["spread_velocity"] < 0.6:  # Caught before it went viral
                    self._early_actions += 1

        # Track successful protection of real users
        elif action == "ALLOW" and not is_fake:
            self._real_posts_protected += 1

        # Track escalations
        elif action == "ESCALATE":
            self._escalated_posts[self._current_post_index] = self._state.step_count
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

    def _make_observation(self, post_index: int, reward: float, done: bool) -> ModerationObservation:
        p_idx = min(post_index, len(self._posts) - 1)
        post, step = self._posts[p_idx], self._state.step_count
        fc = 0.1 if post.get("is_campaign") else min(0.1 + (step * self.config["factcheck_rise_rate"]), 0.95)
        
        warning_msg = "CRITICAL ALERT: Botnet attack." if post.get("is_campaign") else "None"
        uid = post.get("user_id", "unknown")

        return ModerationObservation(
            misinfo_probability=post["misinfo_probability"], 
            virality_score=post["virality_score"],
            spread_velocity=post["spread_velocity"], 
            report_count=min(post["report_count"], 200), 
            trusted_report_count=min(post["trusted_report_count"], 50), 
            reporter_trust=post["reporter_trust"],
            user_credibility=post["user_credibility"], 
            is_repeat_offender=post["is_repeat_offender"],
            factcheck_confidence=round(fc, 3), 
            content_category=post["content_category"],
            step_number=step, 
            posts_remaining=max(0, self.config["total_posts"] - p_idx - 1),
            environmental_warning=warning_msg,
            user_id=uid,
            dynamic_reputation=round(self.user_registry.get(uid, 0.7), 3),
            done=done, reward=reward,
        )

    def _get_current_factcheck(self, post_index: int) -> float:
        return 0.1 if self._posts[post_index].get("is_campaign") else min(0.1 + (self._state.step_count * self.config["factcheck_rise_rate"]), 0.95)

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

class EasyGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try: return max(0.01, min(0.99, float(getattr(env, "unwrapped", env)._grade_task1())))
        except Exception: return 0.01

class MediumGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try: return max(0.01, min(0.99, float(getattr(env, "unwrapped", env)._grade_task2())))
        except Exception: return 0.01

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try: return max(0.01, min(0.99, float(getattr(env, "unwrapped", env)._grade_task3())))
        except Exception: return 0.01