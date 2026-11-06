import os
import sys
import tarfile
import shutil
import importlib
from kaggle_environments import make

def load_agents():
    v14_tar = "submission_v14.tar.gz"
    temp_v14_dir = os.path.abspath("temp_test_v14")
    os.makedirs(temp_v14_dir, exist_ok=True)
    with tarfile.open(v14_tar) as tar:
        tar.extractall(temp_v14_dir)
        
    sys.path.insert(0, temp_v14_dir)
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
    v14_module = importlib.import_module("main")
    v14_agent = v14_module.agent
    sys.path.remove(temp_v14_dir)
    
    # We will temporarily patch V16 to behave like Config 4 (which is V15)
    planner_path = os.path.join("buddy_model", "orbit_lite", "planner_core.py")
    main_path = os.path.join("buddy_model", "main.py")
    
    shutil.copy2(planner_path, planner_path + ".bak")
    shutil.copy2(main_path, main_path + ".bak")
    
    target_pref = """    attack_pref = torch.where(
        attack_mask,
        -proximity + 2.0 * prod - 20.0 * blocked_from_all_sources.float(),
        torch.full_like(proximity, float("-inf"))
    )"""
    replacement_pref = """    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float("-inf")))"""
    
    with open(planner_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(target_pref, replacement_pref)
    with open(planner_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    target_evac = """    else:
        evac_entries = plan_midgame_evacuations(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=status,
            leftover=leftover,
            config=config
        )"""
    replacement_evac = """    else:
        evac_entries = _empty_entries(device, dtype)"""
        
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(target_evac, replacement_evac)
    
    # Disable staging
    target_redirect = "active = active | redirect_mask"
    replacement_redirect = "active = active"
    content = content.replace(target_redirect, replacement_redirect)
    
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    sys.path.insert(0, os.path.abspath("buddy_model"))
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
    v16_module = importlib.import_module("main")
    v15_agent = v16_module.agent
    sys.path.remove(os.path.abspath("buddy_model"))
    
    return v15_agent, v14_agent, temp_v14_dir, planner_path, main_path

def main():
    try:
        v15_agent, v14_agent, temp_dir, planner_path, main_path = load_agents()
    except Exception as e:
        print(f"Error loading agents: {e}")
        return
        
    num_games = 50
    print(f"Running {num_games} matches between V15 (Config 4) and V14...")
    
    v15_wins = 0
    v14_wins = 0
    draws = 0
    errors = 0
    
    for i in range(num_games):
        swap = (i % 2 == 1)
        seed = 4000 + i
        env = make("orbit_wars", debug=False, configuration={"seed": seed})
        agents = [v14_agent, v15_agent] if swap else [v15_agent, v14_agent]
        result = env.run(agents)
        final = result[-1]
        r0, r1 = final[0]['reward'], final[1]['reward']
        if r0 is None or r1 is None:
            errors += 1
            continue
        v15_reward = r1 if swap else r0
        v14_reward = r0 if swap else r1
        if v15_reward > v14_reward:
            v15_wins += 1
        elif v15_reward < v14_reward:
            v14_wins += 1
        else:
            draws += 1
            
    print("\n=== MATCHUP RESULTS (V15 vs V14) ===")
    print(f"Total games: {num_games}")
    print(f"V15 Wins:    {v15_wins} ({v15_wins/num_games*100:.1f}%)")
    print(f"V14 Wins:    {v14_wins} ({v14_wins/num_games*100:.1f}%)")
    print(f"Draws:       {draws}")
    
    # Restore backups
    shutil.copy2(planner_path + ".bak", planner_path)
    shutil.copy2(main_path + ".bak", main_path)
    os.remove(planner_path + ".bak")
    os.remove(main_path + ".bak")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
