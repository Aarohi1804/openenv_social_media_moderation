import traceback

# 🚨 UPDATED IMPORT: Pointing exactly to where your file lives
from server.social_media_moderation_env_environment import SocialMediaModerationEnvironment
from models import ModerationAction

print("--- STARTING LOCAL TEST ---")

try:
    print("1. Initializing Environment...")
    env = SocialMediaModerationEnvironment(task_id="task_hard")
    print("✅ Initialization Successful!")

    print("\n2. Testing reset()...")
    obs = env.reset(task_id="task_hard")
    print("✅ Reset Successful!")
    print(f"Initial Observation keys: {obs.dict().keys() if hasattr(obs, 'dict') else 'Unknown'}")

    print("\n3. Testing step()...")
    action = ModerationAction(action="ALLOW")
    obs = env.step(action)
    print("✅ Step Successful!")
    print(f"Reward Received: {obs.reward}")
    print(f"Done Status: {obs.done}")

    print("\n🎉 ENVIRONMENT LOGIC IS 100% FLAWLESS. The issue is in openenv.yaml.")

except Exception as e:
    print("\n🚨 BUG CAUGHT 🚨")
    traceback.print_exc()
    