from stable_baselines3 import PPO
import numpy as np

print("Extracting weights from final model ppo_orbit_wars.zip...")
model = PPO.load("ppo_orbit_wars.zip")

weights = {}
for k, v in model.policy.state_dict().items():
    if "policy_net" in k or "action_net" in k:
        weights[k] = v.cpu().numpy()

np.savez("ppo_weights.npz", **weights)
print("Saved ppo_weights.npz")
