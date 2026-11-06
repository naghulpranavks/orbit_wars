import os
import sys
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from rl_env import OrbitWarsRLEnv

# ─── Configuration ───────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000       # Full production training loop
SELF_PLAY_SAVE_FREQ = 50_000      # Save to historical_models/ for self-play opponents
CHECKPOINT_SAVE_FREQ = 10_000     # Save recovery checkpoints (to resume if interrupted)
CHECKPOINT_DIR = "checkpoints"
HISTORY_DIR = "historical_models"
FINAL_MODEL_NAME = "ppo_orbit_wars"
USE_WANDB = True                  # Set to False to skip wandb entirely


class SelfPlaySaveCallback(BaseCallback):
    """
    Custom callback for saving the model every N steps to the history folder.
    This enables the environment to load these models for Self-Play.
    """
    def __init__(self, save_freq=50000, history_dir="historical_models", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.history_dir = history_dir

    def _init_callback(self) -> None:
        os.makedirs(self.history_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.history_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)
            print(f"\n[Self-Play] Saved historical model to {model_path}")
        return True


class ProgressCallback(BaseCallback):
    """Prints training progress every N steps."""
    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            print(f"[Progress] Step {self.num_timesteps:,} / {TOTAL_TIMESTEPS:,} "
                  f"({100 * self.num_timesteps / TOTAL_TIMESTEPS:.1f}%)")
        return True


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint to resume training from.
    Searches both checkpoints/ and historical_models/ directories.
    """
    import re
    search_dirs = [checkpoint_dir, HISTORY_DIR]
    best_step = -1
    best_path = None
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".zip"):
                continue
            match = re.search(r'(\d+)', f)
            if match:
                step = int(match.group(1))
                if step > best_step:
                    best_step = step
                    best_path = os.path.join(d, f)
    
    if best_path:
        print(f"[Resume] Found checkpoint at step {best_step:,}: {best_path}")
    return best_path


def init_wandb():
    """Try to initialize wandb. Returns the run object or None if it fails."""
    if not USE_WANDB:
        print("[wandb] Skipped (USE_WANDB=False)")
        return None
    try:
        run = wandb.init(
            project="orbit-wars-self-play",
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )
        print(f"[wandb] Connected! Run URL: {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] WARNING: Failed to initialize ({e})")
        print("[wandb] Training will continue WITHOUT wandb logging.")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("  Orbit Wars - Self-Play PPO Training")
    print("=" * 60)

    # 1. Try wandb (non-blocking)
    wandb_run = init_wandb()

    # 2. Initialize environment
    print("\n[Init] Creating Orbit Wars Self-Play Environment...")
    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make_env():
        def _init():
            return OrbitWarsRLEnv(history_dir=HISTORY_DIR)
        return _init
    
    # Run 8 environments in parallel on the local machine
    env = SubprocVecEnv([make_env() for _ in range(8)])

    # 3. Load or create model
    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"[Init] Resuming from checkpoint: {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=env,
                         tensorboard_log=f"runs/{wandb_run.id if wandb_run else 'local'}")
    else:
        print("[Init] Starting fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            tensorboard_log=f"runs/{wandb_run.id if wandb_run else 'local'}"
        )

    # 4. Build callbacks
    callbacks = [
        # Self-play historical model saving
        SelfPlaySaveCallback(save_freq=SELF_PLAY_SAVE_FREQ, history_dir=HISTORY_DIR, verbose=1),
        # Recovery checkpoints (SB3 built-in)
        CheckpointCallback(save_freq=CHECKPOINT_SAVE_FREQ, save_path=CHECKPOINT_DIR,
                           name_prefix="rl_model", verbose=1),
        # Console progress
        ProgressCallback(log_freq=2048),
    ]

    # Add wandb callback only if connected
    if wandb_run:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{wandb_run.id}",
            verbose=2,
        ))

    # 5. Train!
    print(f"\n[Train] Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"[Train] Checkpoints -> {CHECKPOINT_DIR}/ (every {CHECKPOINT_SAVE_FREQ:,} steps)")
    print(f"[Train] Self-play models -> {HISTORY_DIR}/ (every {SELF_PLAY_SAVE_FREQ:,} steps)")
    print(f"[Train] wandb -> {'ENABLED' if wandb_run else 'DISABLED'}")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=(latest_ckpt is None),  # Continue step count if resuming
        )
    except KeyboardInterrupt:
        print("\n[Train] Interrupted! Saving emergency checkpoint...")
        model.save(os.path.join(CHECKPOINT_DIR, "interrupted_model"))
        print(f"[Train] Saved to {CHECKPOINT_DIR}/interrupted_model.zip")

    # 6. Save final model
    print("\n[Done] Training complete! Saving final model...")
    model.save(FINAL_MODEL_NAME)
    print(f"[Done] Final model saved to {FINAL_MODEL_NAME}.zip")

    if wandb_run:
        wandb.finish()

    # Summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Final model:      {FINAL_MODEL_NAME}.zip")
    print(f"  Checkpoints:      {CHECKPOINT_DIR}/")
    print(f"  Self-play models: {HISTORY_DIR}/")
    if wandb_run:
        print(f"  wandb dashboard:  {wandb_run.url}")
    print("=" * 60)
