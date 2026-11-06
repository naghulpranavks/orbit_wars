import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make
import sys
import os

sys.path.append(os.path.abspath('buddy_model'))
import main as buddy_main
from orbit_lite.obs import parse_obs
from orbit_lite.adapter import single_obs_to_tensor
import torch

class RLTunerEnv(gym.Env):
    """
    V9 RL Environment: The AI doesn't control ships. 
    It controls the PyTorch mathematical parameters every single turn.
    """
    def __init__(self):
        super(RLTunerEnv, self).__init__()
        # Actions: [horizon, max_waves, roi_threshold, min_ships] mapped from [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Obs: [step, my_planets, enemy_planets, my_ships, enemy_ships]
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(5,), dtype=np.float32)
        self.env = make("orbit_wars", configuration={"episodeSteps": 500})
        self.trainer = None
        self.obs = None
        self.player_id = 0

    def reset(self, seed=None, options=None):
        # OPPONENT POOL TRAINING: Randomly select an opponent type every match!
        import random
        
        # We use a mix of Random (unpredictable chaos) and V7 PyTorch (elite logic)
        opponent_pool = ["random", buddy_main.agent]
        selected_opponent = random.choice(opponent_pool)
        
        self.trainer = self.env.train([None, selected_opponent])
        self.obs = self.trainer.reset()
        self.player_id = self.obs.get("player", 0) if isinstance(self.obs, dict) else self.obs.player
        buddy_main._RUNTIME.reset()
        return self._get_obs(self.obs), {}

    def _get_obs(self, obs_raw):
        step = obs_raw.get("step", 0) if isinstance(obs_raw, dict) else obs_raw.step
        raw_planets = obs_raw.get("planets", []) if isinstance(obs_raw, dict) else obs_raw.planets
        
        my_p = sum(1 for p in raw_planets if int(p[1]) == self.player_id)
        en_p = sum(1 for p in raw_planets if int(p[1]) not in [-1, self.player_id])
        my_s = sum(float(p[5]) for p in raw_planets if int(p[1]) == self.player_id)
        en_s = sum(float(p[5]) for p in raw_planets if int(p[1]) not in [-1, self.player_id])
        
        return np.array([step/500.0, my_p/60.0, en_p/60.0, my_s/1000.0, en_s/1000.0], dtype=np.float32)

    def step(self, action):
        # 1. Scale AI Actions back into PyTorch math parameters
        h = int(np.interp(action[0], [-1, 1], [18, 25]))
        mw = int(np.interp(action[1], [-1, 1], [6, 15]))
        roi = float(np.interp(action[2], [-1, 1], [1.0, 3.0]))
        ms = float(np.interp(action[3], [-1, 1], [2.0, 10.0]))

        class TunedConfig:
            horizon: int = h
            max_sources_per_lane: int = 12
            max_offensive_targets: int = 12
            max_defensive_targets: int = 4
            max_waves_per_turn: int = mw
            roi_threshold: float = roi
            min_ships_to_launch: float = ms
            enable_regroup: bool = True
            max_regroup_time: float = 7.0
            regroup_pressure_delta_min: float = 0.25
            max_regroup_sources_per_lane: int = 6
            max_regroup_targets_per_source: int = 7
            regroup_pressure_norm: str = "none"
            regroup_time_penalty_weight: float = 1e-3

        # 2. Let PyTorch Engine do all the exact math using the AI's settings
        obs_tensors = single_obs_to_tensor(self.obs, player_id=self.player_id)
        
        with torch.no_grad():
            sparse_row = buddy_main.run_turn(
                obs_tensors,
                config=TunedConfig,
                player_count=2,
                memory=buddy_main._RUNTIME.memory
            )
            
        moves = buddy_main.sparse_action_row_to_moves(sparse_row, self.obs, player_id=self.player_id)

        # 3. Apply moves to Kaggle Board
        self.obs, reward, done, info = self.trainer.step(moves)
        
        if isinstance(reward, (list, tuple)):
            reward = reward[0]
            
        if reward is None:
            reward = 0.0

        # Reward Shaping: Give AI tiny rewards for gaining ship advantage, huge reward for winning
        obs_vec = self._get_obs(self.obs)
        ship_advantage = obs_vec[3] - obs_vec[4] 
        shaped_reward = (reward * 10.0) + (ship_advantage * 0.01)

        terminated = done
        truncated = False
        return obs_vec, float(shaped_reward), terminated, truncated, {}
