import math
import os
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from rl_env import OrbitWarsRLEnv

weights = None
env_wrapper = None

def relu(x):
    return np.maximum(0, x)

def agent(obs):
    global weights
    global env_wrapper
    
    if weights is None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = "/kaggle_simulations/agent/"
            
        weights = dict(np.load(os.path.join(current_dir, "ppo_weights.npz")))
        env_wrapper = OrbitWarsRLEnv()
        
    obs_matrix, _ = env_wrapper._get_obs(obs)
    # Flatten obs (60 x 69 = 4140)
    x = obs_matrix.flatten()
    
    # Layer 0
    w0 = weights["mlp_extractor.policy_net.0.weight"]
    b0 = weights["mlp_extractor.policy_net.0.bias"]
    x = relu(np.dot(w0, x) + b0)
    
    # Layer 2
    w2 = weights["mlp_extractor.policy_net.2.weight"]
    b2 = weights["mlp_extractor.policy_net.2.bias"]
    x = relu(np.dot(w2, x) + b2)
    
    # Action net
    w_out = weights["action_net.weight"]
    b_out = weights["action_net.bias"]
    logits = np.dot(w_out, x) + b_out
    
    # The logits array has size 3600 (60 planets * 60 targets)
    action = []
    offset = 0
    for _ in range(env_wrapper.MAX_PLANETS):
        target_logits = logits[offset : offset + 60]
        target_idx = np.argmax(target_logits)
        offset += 60
        action.append(target_idx)
        
    # Hybrid Math Heuristic Decoding
    kaggle_action = []
    raw_planets = obs.get("planets", [])
    my_player_id = obs.get("player", 0)
    
    for src_idx in range(env_wrapper.MAX_PLANETS):
        if src_idx >= len(raw_planets):
            break
            
        target_idx = action[src_idx]
        src_p = Planet(*raw_planets[src_idx])
        
        if src_p.owner == my_player_id and target_idx < len(raw_planets) and target_idx != src_idx:
            target_p = Planet(*raw_planets[target_idx])
            d = math.ceil(math.hypot(target_p.x - src_p.x, target_p.y - src_p.y))
            
            if target_p.owner == my_player_id:
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
                    
    return kaggle_action
