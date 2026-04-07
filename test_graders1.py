import random
import sys
import os

# Ensure Python can see the 'server' directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.social_media_moderation_env_environment import (
    SocialMediaModerationEnvironment,
    EasyGrader,
    MediumGrader,
    HardGrader
)

# Mock the action object to bypass the import error!
class MockAction:
    def __init__(self, action_str):
        self.action = action_str

def test_environment():
    tasks = [
        ("task_easy", EasyGrader),
        ("task_medium", MediumGrader),
        ("task_hard", HardGrader)
    ]

    for task_id, GraderClass in tasks:
        print(f"\n{'='*40}")
        print(f"🛠️ TESTING: {task_id}")
        print(f"{'='*40}")
        
        try:
            # 1. Initialize the Environment
            env = SocialMediaModerationEnvironment(task_id=task_id)
            env.reset()
            print("✅ Environment initialized and reset successfully.")

            # 2. Run a simulation (5 random actions)
            done = False
            steps = 0
            while not done and steps < 5:
                # Pick a random valid action
                action_str = random.choice(["ALLOW", "LABEL_WARNING", "REDUCE_REACH", "DELETE", "ESCALATE"])
                action = MockAction(action_str)
                
                obs = env.step(action)
                done = obs.done
                steps += 1
                
                # Verify step reward is clamped
                assert 0.01 <= obs.reward <= 0.99, f"Step reward {obs.reward} is out of bounds!"

            print(f"✅ Simulated {steps} steps. Step rewards are within [0.01, 0.99].")

            # 3. Test the specific Grader Class
            grader = GraderClass()
            final_score = grader.grade(env)
            
            print(f"📊 Raw Grader Output: {final_score}")

            # 4. Final Validation
            if 0.01 <= final_score <= 0.99:
                print("✅ FINAL SCORE IS VALID! [0.01 <= score <= 0.99]")
            else:
                print("❌ ERROR: FINAL SCORE IS OUT OF BOUNDS!")

        except Exception as e:
            print(f"❌ CRASH ON {task_id}: {str(e)}")

if __name__ == "__main__":
    test_environment()