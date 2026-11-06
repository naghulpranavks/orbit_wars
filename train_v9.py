import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_tuner_env import RLTunerEnv

def train():
    print("==================================================")
    print("V9 DYNAMIC RL HYPER-TUNER: AI Commander Training")
    print("==================================================")
    
    # 1. Initialize Environment
    env = RLTunerEnv()
    print("Checking Custom Environment...")
    check_env(env)
    
    # 2. Build the AI Commander
    # We use a very tiny neural network ([64, 64]) because its only job is
    # to output 4 numbers. It doesn't need to learn complex physics.
    print("Initializing PPO Neural Network...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=2048, 
        batch_size=64,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    
    # 3. Train the AI Commander
    print("Starting Training (Playing against Kaggle Baseline...)")
    time.sleep(1)
    
    # Train for 50,000 timesteps (which translates to about 100 matches)
    model.learn(total_timesteps=50000)
    
    # 4. Save the Model
    model.save("v9_ai_commander")
    print("==================================================")
    print("Training Complete! Saved as v9_ai_commander.zip")
    print("Next step: Package submission_v9.tar.gz")

if __name__ == "__main__":
    train()
