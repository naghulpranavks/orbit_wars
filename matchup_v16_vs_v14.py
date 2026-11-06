import os
import sys
import tarfile
import shutil
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle_environments import make

def load_agent_functions():
    v16_path_abs = os.path.abspath("buddy_model")
    v14_tar = "submission_v14.tar.gz"
    temp_dir = os.path.abspath("temp_v14")
    
    # 1. Extract V14
    os.makedirs(temp_dir, exist_ok=True)
    with tarfile.open(v14_tar) as tar:
        tar.extractall(temp_dir)
        
    # 2. Import V16
    sys.path.insert(0, v16_path_abs)
    # Clear any existing modules to force fresh import
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
            
    v16_module = importlib.import_module("main")
    v16_agent = v16_module.agent
    sys.path.remove(v16_path_abs)
    
    # 3. Import V14
    sys.path.insert(0, temp_dir)
    # Clear modules again so it doesn't use V16's orbit_lite
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
            
    v14_module = importlib.import_module("main")
    v14_agent = v14_module.agent
    sys.path.remove(temp_dir)
    
    # Force clean modules one last time to prevent pollution
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
            
    return v16_agent, v14_agent, temp_dir

def run_one_game(seed, v16_agent, v14_agent, swap_players=False):
    env = make("orbit_wars", debug=True, configuration={"seed": seed})
    
    if not swap_players:
        agents = [v16_agent, v14_agent]
    else:
        agents = [v14_agent, v16_agent]
        
    result = env.run(agents)
    final = result[-1]
    r0 = final[0]['reward']
    r1 = final[1]['reward']
    status0 = final[0]['status']
    status1 = final[1]['status']
    
    if status0 == 'ERROR' or status1 == 'ERROR':
        return 'ERROR', 0, 0
        
    if not swap_players:
        v16_reward = r0
        v14_reward = r1
    else:
        v16_reward = r1
        v14_reward = r0
        
    if v16_reward > v14_reward:
        return 'V16_WIN', v16_reward, v14_reward
    elif v16_reward < v14_reward:
        return 'V14_WIN', v16_reward, v14_reward
    else:
        return 'DRAW', v16_reward, v14_reward

def main():
    try:
        v16_agent, v14_agent, temp_dir = load_agent_functions()
    except Exception as e:
        print(f"Error loading agents: {e}")
        return
        
    num_games = 50
    print(f"Running {num_games} matches between V16 and V14 in-process...")
    
    v16_wins = 0
    v14_wins = 0
    draws = 0
    errors = 0
    
    # Run games sequentially to avoid race conditions on global state/modules
    for i in range(num_games):
        swap = (i % 2 == 1)
        seed = 2000 + i
        print(f"Running game {i} (seed {seed}, swap={swap})...", end="", flush=True)
        res, r16, r14 = run_one_game(seed, v16_agent, v14_agent, swap)
        if res == 'V16_WIN':
            v16_wins += 1
            print(f" V16 Win (V16={r16}, V14={r14})", flush=True)
        elif res == 'V14_WIN':
            v14_wins += 1
            print(f" V14 Win (V16={r16}, V14={r14})", flush=True)
        elif res == 'DRAW':
            draws += 1
            print(f" Draw (V16={r16}, V14={r14})", flush=True)
        else:
            errors += 1
            print(" ERROR", flush=True)
            
    print("\n=== DIRECT MATCHUP RESULTS (V16 vs V14) ===", flush=True)
    print(f"Total games: {num_games}", flush=True)
    print(f"V16 Wins:    {v16_wins} ({v16_wins/num_games*100:.1f}%)", flush=True)
    print(f"V14 Wins:    {v14_wins} ({v14_wins/num_games*100:.1f}%)", flush=True)
    print(f"Draws:       {draws}", flush=True)
    print(f"Errors:      {errors}", flush=True)
    
    # Clean up
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()
