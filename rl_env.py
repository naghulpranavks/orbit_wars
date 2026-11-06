import os
import glob
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class OrbitWarsRLEnv(gym.Env):
    """
    A Gymnasium wrapper for Orbit Wars that supports Self-Play.
    V3 "Master Tier" Architecture with Spatial Matrix and Hybrid Math Heuristic.
    """
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, history_dir="historical_models"):
        super().__init__()
        self.env = make("orbit_wars", configuration={"seed": 42}, debug=True)
        self.history_dir = history_dir
        self.trainer = None
        
        self.MAX_PLANETS = 60
        self.MAX_SHIPS_NORM = 2000.0
        
        # Spatial Awareness Matrix: Base 9 features + 60 distances
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.MAX_PLANETS, self.MAX_PLANETS + 9), 
            dtype=np.float32
        )
        
        # Hybrid Heuristic: AI only selects a target_idx (0 to 59) for each planet
        self.action_space = spaces.MultiDiscrete(
            np.array([self.MAX_PLANETS] * self.MAX_PLANETS).flatten()
        )
        
        self.last_obs = None
        self.last_total_ships = 0
        self.my_player_id = 0
        self.planet_id_to_idx = {}
        
        # Cache for loaded models so we don't spam disk I/O
        self.loaded_models_cache = {}

    def _get_obs(self, obs):
        self.my_player_id = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        comet_ids = obs.get("comet_planet_ids", [])
        raw_fleets = obs.get("fleets", [])
        
        # Calculate incoming ships per planet ID
        from collections import defaultdict
        incoming_friendly = defaultdict(int)
        incoming_enemy = defaultdict(int)
        
        for f in raw_fleets:
            owner = f[1]
            num_ships = f[5]
            target_id = f[6]
            if owner == self.my_player_id:
                incoming_friendly[target_id] += num_ships
            else:
                incoming_enemy[target_id] += num_ships
        
        obs_matrix = np.zeros((self.MAX_PLANETS, self.MAX_PLANETS + 9), dtype=np.float32)
        
        # Precompute Spatial Awareness (Distance Matrix)
        dist_matrix = np.zeros((self.MAX_PLANETS, self.MAX_PLANETS), dtype=np.float32)
        for i in range(len(raw_planets)):
            if i >= self.MAX_PLANETS: break
            p1 = Planet(*raw_planets[i])
            for j in range(len(raw_planets)):
                if j >= self.MAX_PLANETS: break
                p2 = Planet(*raw_planets[j])
                d = math.hypot(p2.x - p1.x, p2.y - p1.y)
                dist_matrix[i][j] = d / 200.0  # Normalize distance
                
        self.planet_id_to_idx = {}
        total_my_ships = 0
        
        for i, p_data in enumerate(raw_planets):
            if i >= self.MAX_PLANETS: break
            p = Planet(*p_data)
            self.planet_id_to_idx[i] = p.id
            
            owner_status = 0.0
            if p.owner == self.my_player_id:
                owner_status = 1.0
                total_my_ships += p.ships
            elif p.owner != -1:
                owner_status = -1.0
                
            base_features = [
                owner_status, 
                p.x / 100.0, 
                p.y / 100.0, 
                min(p.radius / 50.0, 1.0), 
                min(p.ships / self.MAX_SHIPS_NORM, 1.0), 
                p.production / 5.0, 
                1.0 if p.id in comet_ids else 0.0,
                min(incoming_friendly[p.id] / self.MAX_SHIPS_NORM, 1.0),
                min(incoming_enemy[p.id] / self.MAX_SHIPS_NORM, 1.0)
            ]
            
            obs_matrix[i][:9] = base_features
            obs_matrix[i][9:] = dist_matrix[i]
            
        return obs_matrix, total_my_ships

    def _make_historical_agent(self, model_path):
        """Creates a Kaggle-compatible agent function using a saved PPO model."""
        from stable_baselines3 import PPO
        if model_path not in self.loaded_models_cache:
            self.loaded_models_cache[model_path] = PPO.load(model_path)
        model = self.loaded_models_cache[model_path]
        
        def agent(obs):
            # Parse observation for the model
            obs_matrix, _ = self._get_obs(obs)
            action, _ = model.predict(obs_matrix, deterministic=True)
            
            kaggle_action = []
            raw_planets = obs.get("planets", [])
            my_id = obs.get("player", 0)
            
            for src_idx in range(self.MAX_PLANETS):
                if src_idx >= len(raw_planets):
                    break
                
                target_idx = action[src_idx]
                
                if target_idx < len(raw_planets) and target_idx != src_idx:
                    src_p = Planet(*raw_planets[src_idx])
                    target_p = Planet(*raw_planets[target_idx])
                    
                    if src_p.owner == my_id:
                        d = math.ceil(math.hypot(target_p.x - src_p.x, target_p.y - src_p.y))
                        
                        if target_p.owner == my_id:
                            # Friendly: send 50%
                            num_ships = int(src_p.ships * 0.5)
                        else:
                            # Enemy/Neutral: send exact amount needed
                            prod = target_p.production if target_p.owner != -1 else 0
                            ships_needed = target_p.ships + (prod * d) + 1
                            if src_p.ships >= ships_needed:
                                num_ships = ships_needed
                            else:
                                num_ships = 0
                                
                        if num_ships > 0:
                            dy = target_p.y - src_p.y
                            dx = target_p.x - src_p.x
                            angle = math.atan2(dy, dx)
                            kaggle_action.append([src_p.id, angle, num_ships])
                            
            return kaggle_action
            
        return agent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.env.configuration["seed"] = seed
             
        # SELF-PLAY DYNAMIC MATCHMAKING
        os.makedirs(self.history_dir, exist_ok=True)
        historical_models = glob.glob(os.path.join(self.history_dir, "*.zip"))
        
        if len(historical_models) == 0:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            greedy_agent_path = os.path.join(current_dir, "agent.py")
            self.trainer = self.env.train([None, greedy_agent_path])
        else:
            if random.random() < 0.8:
                selected_model = max(historical_models, key=os.path.getctime)
            else:
                selected_model = random.choice(historical_models)
                
            historical_agent_fn = self._make_historical_agent(selected_model)
            self.trainer = self.env.train([None, historical_agent_fn])
             
        obs = self.trainer.reset()
        self.last_obs = obs
        obs_matrix, total_my_ships = self._get_obs(obs)
        self.last_total_ships = total_my_ships
        return obs_matrix, {}

    def step(self, action):
        kaggle_action = []
        raw_planets = self.last_obs.get("planets", [])
        
        penalty = 0.0
        
        for src_idx in range(self.MAX_PLANETS):
            if src_idx >= len(raw_planets):
                break
            
            target_idx = action[src_idx]
            src_p = Planet(*raw_planets[src_idx])
            
            # Action Masking (Negative Reinforcement)
            if src_p.owner != self.my_player_id:
                if target_idx != src_idx:
                    penalty -= 0.1 # Penalty for trying to command an enemy planet
                continue
                
            if target_idx < len(raw_planets) and target_idx != src_idx:
                target_p = Planet(*raw_planets[target_idx])
                
                # Hybrid Math Heuristic
                d = math.ceil(math.hypot(target_p.x - src_p.x, target_p.y - src_p.y))
                
                if target_p.owner == self.my_player_id:
                    num_ships = int(src_p.ships * 0.5)
                else:
                    prod = target_p.production if target_p.owner != -1 else 0
                    ships_needed = target_p.ships + (prod * d) + 1
                    
                    if src_p.ships >= ships_needed:
                        num_ships = ships_needed
                    else:
                        num_ships = 0
                        
                if num_ships > 0:
                    dy = target_p.y - src_p.y
                    dx = target_p.x - src_p.x
                    angle = math.atan2(dy, dx)
                    kaggle_action.append([src_p.id, angle, num_ships])
        
        # Count our planets before the step
        prev_planet_count = sum(1 for p in raw_planets if Planet(*p).owner == self.my_player_id)
        
        next_obs, reward_from_env, done, info = self.trainer.step(kaggle_action)
        self.last_obs = next_obs
        obs_matrix, total_my_ships = self._get_obs(next_obs)
        
        # Count our planets after the step
        new_raw_planets = next_obs.get("planets", [])
        new_planet_count = sum(1 for p in new_raw_planets if Planet(*p).owner == self.my_player_id)
        
        # Base reward: change in total ships
        ship_delta = (total_my_ships - self.last_total_ships) / 100.0 
        self.last_total_ships = total_my_ships
        
        # Reward shaping + Action Penalties
        planet_delta = new_planet_count - prev_planet_count
        reward = ship_delta + (planet_delta * 0.5) + penalty
        
        if done:
            if reward_from_env == 1.0: reward += 10.0
            elif reward_from_env == -1.0: reward -= 10.0
                
        return obs_matrix, reward, done, False, {}

    def render(self):
        return self.env.render(mode="ansi")
