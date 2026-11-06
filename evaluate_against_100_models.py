import os
import sys
import glob
import re
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

# Add current directory to path so we can import local modules
sys.path.append(os.path.abspath('.'))
from rl_env import OrbitWarsRLEnv

def make_adapted_agent(model_path, env_wrapper):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    
    # Detect expected shape of the input layer
    expected_shape = model.observation_space.shape # e.g. (60, 7)
    num_features = expected_shape[1]
    
    def agent(obs):
        # 1. Get the full (60, 69) observation matrix
        obs_matrix, _ = env_wrapper._get_obs(obs)
        # 2. Slice it to match the expected number of features
        sliced_obs = obs_matrix[:, :num_features]
        # 3. Predict action
        action, _ = model.predict(sliced_obs, deterministic=True)
        
        # 4. Decode action based on action space shape
        kaggle_action = []
        raw_planets = obs.get("planets", [])
        my_id = obs.get("player", 0)
        
        # Case A: MultiDiscrete([60, 60, 11])
        if len(action) == 3:
            src_idx, tgt_idx, ship_code = action
            if src_idx < len(raw_planets) and tgt_idx < len(raw_planets) and src_idx != tgt_idx:
                src_p = Planet(*raw_planets[src_idx])
                tgt_p = Planet(*raw_planets[tgt_idx])
                if src_p.owner == my_id:
                    num_ships = int(src_p.ships * (ship_code / 10.0))
                    if num_ships > 0:
                        dy = tgt_p.y - src_p.y
                        dx = tgt_p.x - src_p.x
                        kaggle_action.append([src_p.id, math.atan2(dy, dx), num_ships])
            return kaggle_action
            
        # Case B: MultiDiscrete([60, 11] * 60)
        elif len(action) == 120:
            for src_idx in range(60):
                if src_idx >= len(raw_planets):
                    break
                target_idx = action[src_idx * 2]
                ship_code = action[src_idx * 2 + 1]
                if target_idx < len(raw_planets) and target_idx != src_idx:
                    src_p = Planet(*raw_planets[src_idx])
                    tgt_p = Planet(*raw_planets[target_idx])
                    if src_p.owner == my_id:
                        num_ships = int(src_p.ships * (ship_code / 10.0))
                        if num_ships > 0:
                            dy = tgt_p.y - src_p.y
                            dx = tgt_p.x - src_p.x
                            kaggle_action.append([src_p.id, math.atan2(dy, dx), num_ships])
            return kaggle_action
            
        # Case C: MultiDiscrete([60] * 60)
        else:
            for src_idx in range(60):
                if src_idx >= len(raw_planets):
                    break
                target_idx = action[src_idx]
                if target_idx < len(raw_planets) and target_idx != src_idx:
                    src_p = Planet(*raw_planets[src_idx])
                    target_p = Planet(*raw_planets[target_idx])
                    if src_p.owner == my_id:
                        d = math.ceil(math.hypot(target_p.x - src_p.x, target_p.y - src_p.y))
                        if target_p.owner == my_id:
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
                            kaggle_action.append([src_p.id, math.atan2(dy, dx), num_ships])
            return kaggle_action
            
    return agent

def main():
    history_dir = "historical_models_v1"
    v14_path = os.path.join("buddy_model", "main.py")
    
    # 1. Find and sort all historical models
    zip_files = glob.glob(os.path.join(history_dir, "*.zip"))
    models_with_steps = []
    for f in zip_files:
        match = re.search(r'model_step_(\d+)\.zip', os.path.basename(f))
        if match:
            step = int(match.group(1))
            models_with_steps.append((step, f))
    
    models_with_steps.sort(key=lambda x: x[0])
    
    # Take 100 models (specifically, let's take every step or a spread of 100 models)
    models_to_test = models_with_steps[:100]
    num_models = len(models_to_test)
    print(f"Found {len(models_with_steps)} historical models in {history_dir}.")
    print(f"Testing V14 against exactly {num_models} checkpoints in sequential order from earliest to latest.")
    
    # Instantiate the env wrapper to use its _get_obs method
    print("Initializing environment wrapper...")
    rl_env_wrapper = OrbitWarsRLEnv()
    
    results = {}
    
    def run_one_match(step, model_path):
        env = make("orbit_wars", debug=True)
        # Create the agent function for this model
        opp_agent = make_adapted_agent(model_path, rl_env_wrapper)
        
        start_time = time.time()
        # V14 is player 0, RL agent is player 1
        result = env.run([v14_path, opp_agent])
        elapsed = time.time() - start_time
        
        final = result[-1]
        r0 = final[0]['reward']
        r1 = final[1]['reward']
        status = final[0]['status']
        
        if status == 'ERROR':
            res = 'ERROR'
        elif r0 > r1:
            res = 'WIN'
        elif r0 < r1:
            res = 'LOSS'
        else:
            res = 'DRAW'
            
        return step, res, r0, r1, elapsed

    print(f"Starting parallel evaluation against {num_models} checkpoints (using 4 threads)...")
    
    start_all = time.time()
    # Use ThreadPoolExecutor to run matches in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_one_match, step, path): (step, path) for step, path in models_to_test}
        
        completed_count = 0
        for future in as_completed(futures):
            try:
                step, res, r0, r1, elapsed = future.result()
                results[step] = (res, r0, r1, elapsed)
            except Exception as e:
                # Store the error
                step_val = futures[future][0]
                results[step_val] = ('ERROR', 0, 0, 0)
                print(f"Error running match for step {step_val}: {e}")
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == num_models:
                print(f"Completed {completed_count}/{num_models} games...")
                
    elapsed_all = time.time() - start_all
    print(f"\nFinished all {num_models} matches in {elapsed_all:.1f} seconds.")
    
    # Analyze and print results
    wins = 0
    losses = 0
    draws = 0
    errors = 0
    
    # Also write detailed log to a text file for record keeping
    out_file_path = "evaluation_100_models_report.txt"
    with open(out_file_path, "w") as out_f:
        header = f"{'Step':<12} | {'Result':<8} | {'V14 Score':<10} | {'Opp Score':<10} | {'Duration':<8}\n"
        separator = "-" * 80 + "\n"
        out_f.write("=" * 80 + "\n")
        out_f.write("DETAILED RESULTS (V14 vs PPO Checkpoints)\n")
        out_f.write("=" * 80 + "\n")
        out_f.write(header)
        out_f.write(separator)
        
        print("\n" + "=" * 80)
        print("DETAILED RESULTS (V14 vs PPO Checkpoints)")
        print("=" * 80)
        print(f"{'Step':<12} | {'Result':<8} | {'V14 Score':<10} | {'Opp Score':<10} | {'Duration':<8}")
        print("-" * 80)
        
        for step, path in models_to_test:
            res, r0, r1, elapsed = results[step]
            line = f"{step:<12,d} | {res:<8} | {r0:<10.1f} | {r1:<10.1f} | {elapsed:.1f}s"
            print(line)
            out_f.write(line + "\n")
            
            if res == 'WIN':
                wins += 1
            elif res == 'LOSS':
                losses += 1
            elif res == 'DRAW':
                draws += 1
            else:
                errors += 1
                
        win_rate = (wins / num_models) * 100
        summary_lines = [
            "=" * 80,
            "SUMMARY:",
            f"  Total games: {num_models}",
            f"  Wins:        {wins} ({win_rate:.1f}%)",
            f"  Losses:      {losses}",
            f"  Draws:       {draws}",
            f"  Errors:      {errors}",
            f"  Total time:  {elapsed_all:.1f}s",
            "=" * 80
        ]
        
        for s_line in summary_lines:
            print(s_line)
            out_f.write(s_line + "\n")
            
    print(f"\nDetailed report saved to {out_file_path}")

if __name__ == "__main__":
    main()
