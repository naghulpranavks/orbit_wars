"""
Kaggle Training Script - Bulletproof version.
Run this directly as a notebook cell (paste the content), NOT as !python external_script.py.
This ensures all files are saved to /kaggle/working/ and appear in the Output tab.
"""
import os
import sys
import math
import glob
import random
import shutil
import numpy as np

# ─── FORCE working directory to /kaggle/working/ ───
KAGGLE_WORKING = "/kaggle/working"
if os.path.exists(KAGGLE_WORKING):
    os.chdir(KAGGLE_WORKING)
    print(f"[Setup] Working directory: {os.getcwd()}")

# ─── Add dataset code to Python path ───
DATASET_CODE_DIR = "/kaggle/input/orbit-wars-self-play-code"
if os.path.exists(DATASET_CODE_DIR):
    sys.path.insert(0, DATASET_CODE_DIR)
    print(f"[Setup] Added {DATASET_CODE_DIR} to sys.path")

# ─── Configuration ───
TOTAL_TIMESTEPS = 2_000_000
SELF_PLAY_SAVE_FREQ = 50_000
CHECKPOINT_SAVE_FREQ = 25_000       # Save recovery checkpoints more often
CHECKPOINT_DIR = os.path.join(KAGGLE_WORKING, "checkpoints")
HISTORY_DIR = os.path.join(KAGGLE_WORKING, "historical_models")
FINAL_MODEL_NAME = os.path.join(KAGGLE_WORKING, "ppo_orbit_wars")

# ─── Create directories with ABSOLUTE paths ───
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
print(f"[Setup] Checkpoint dir: {CHECKPOINT_DIR}")
print(f"[Setup] History dir: {HISTORY_DIR}")

# ─── Verify we can write ───
test_file = os.path.join(KAGGLE_WORKING, "_write_test.txt")
with open(test_file, "w") as f:
    f.write("write test")
assert os.path.exists(test_file), "CANNOT WRITE TO /kaggle/working/!"
os.remove(test_file)
print("[Setup] Write test PASSED ✓")

# ─── Imports ───
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_env import OrbitWarsRLEnv

# ─── wandb (optional) ───
USE_WANDB = True
wandb_run = None
try:
    if USE_WANDB:
        import wandb
        from kaggle_secrets import UserSecretsClient
        secret = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=secret)
        wandb_run = wandb.init(
            project="orbit-wars-self-play",
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )
        print(f"[wandb] Connected! URL: {wandb_run.url}")
except Exception as e:
    print(f"[wandb] Failed: {e}. Continuing without wandb.")
    wandb_run = None


class SelfPlaySaveCallback(BaseCallback):
    """Save model to history dir for self-play opponents + verify the save."""
    def __init__(self, save_freq=50000, history_dir="historical_models", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.history_dir = history_dir

    def _init_callback(self):
        os.makedirs(self.history_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.history_dir, f"model_step_{self.num_timesteps}")
            self.model.save(model_path)
            # VERIFY the save actually worked
            expected_file = model_path + ".zip"
            if os.path.exists(expected_file):
                size_kb = os.path.getsize(expected_file) / 1024
                print(f"\n[Self-Play] ✓ SAVED & VERIFIED: {expected_file} ({size_kb:.0f} KB)")
            else:
                print(f"\n[Self-Play] ✗ SAVE FAILED: {expected_file} does not exist!")
        return True


class ProgressAndSnapshotCallback(BaseCallback):
    """Print progress + periodically save a 'latest' snapshot for safety."""
    def __init__(self, snapshot_freq=10000, verbose=0):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
    
    def _on_step(self):
        if self.n_calls % self.snapshot_freq == 0:
            # Save a "latest" model that gets overwritten each time
            latest_path = os.path.join(KAGGLE_WORKING, "ppo_orbit_wars_latest")
            self.model.save(latest_path)
            
            # Print progress
            pct = 100 * self.num_timesteps / TOTAL_TIMESTEPS
            n_files = len(glob.glob(os.path.join(HISTORY_DIR, "*.zip")))
            working_files = len(glob.glob(os.path.join(KAGGLE_WORKING, "*.zip")))
            print(f"[Progress] Step {self.num_timesteps:,}/{TOTAL_TIMESTEPS:,} ({pct:.1f}%) | "
                  f"History models: {n_files} | Working dir zips: {working_files}")
        return True


# ─── Find checkpoint to resume from ───
def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not checkpoints:
        # Also check historical models
        checkpoints = glob.glob(os.path.join(HISTORY_DIR, "*.zip"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getctime)
    print(f"[Resume] Found checkpoint: {latest}")
    return latest


# ─── Initialize model ───
print("\n" + "=" * 60)
print("  Orbit Wars - Self-Play PPO Training (Kaggle)")
print("=" * 60)

def make_env():
    def _init():
        return OrbitWarsRLEnv(history_dir=HISTORY_DIR)
    return _init

# Run 4 environments in parallel!
env = SubprocVecEnv([make_env() for _ in range(4)])
latest_ckpt = find_latest_checkpoint()

tb_log_dir = os.path.join(KAGGLE_WORKING, "runs", wandb_run.id if wandb_run else "local")

if latest_ckpt:
    print(f"[Init] Resuming from: {latest_ckpt}")
    model = PPO.load(latest_ckpt, env=env, tensorboard_log=tb_log_dir)
else:
    print("[Init] Starting fresh PPO model...")
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=0.0003, n_steps=2048, batch_size=64,
        tensorboard_log=tb_log_dir
    )

# ─── Callbacks ───
callbacks = [
    SelfPlaySaveCallback(save_freq=SELF_PLAY_SAVE_FREQ, history_dir=HISTORY_DIR, verbose=1),
    CheckpointCallback(save_freq=CHECKPOINT_SAVE_FREQ, save_path=CHECKPOINT_DIR,
                       name_prefix="rl_model", verbose=1),
    ProgressAndSnapshotCallback(snapshot_freq=10000),
]

if wandb_run:
    from wandb.integration.sb3 import WandbCallback
    callbacks.append(WandbCallback(
        gradient_save_freq=1000,
        model_save_path=os.path.join(KAGGLE_WORKING, "models", wandb_run.id),
        verbose=2,
    ))

# ─── Train ───
print(f"\n[Train] Starting for {TOTAL_TIMESTEPS:,} timesteps...")
print(f"[Train] All saves go to: {KAGGLE_WORKING}")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=(latest_ckpt is None),
    )
except KeyboardInterrupt:
    print("\n[!] Interrupted!")
except Exception as e:
    print(f"\n[!] Training error: {e}")
finally:
    # ALWAYS save, even on crash/timeout
    print("\n[Final] Saving final model...")
    model.save(FINAL_MODEL_NAME)
    
    # Verify final save
    final_file = FINAL_MODEL_NAME + ".zip"
    if os.path.exists(final_file):
        size_mb = os.path.getsize(final_file) / (1024 * 1024)
        print(f"[Final] ✓ SAVED: {final_file} ({size_mb:.1f} MB)")
    else:
        print(f"[Final] ✗ FAILED to save final model!")
    
    # Print everything in /kaggle/working/
    print(f"\n[Final] Files in {KAGGLE_WORKING}:")
    for root, dirs, files in os.walk(KAGGLE_WORKING):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, KAGGLE_WORKING)
            print(f"  {rel} ({size / 1024:.0f} KB)")
    
    if wandb_run:
        wandb.finish()

print("\n[DONE] Training complete!")
