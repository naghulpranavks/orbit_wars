import os
import sys
import tarfile
import shutil
import importlib
from kaggle_environments import make

# We will dynamically modify planner_core.py or main.py and test:
# Config 1: Pure V16 (current)
# Config 2: V16 with V14 target shortlist (pure -proximity)
# Config 3: V16 with V14 target shortlist + NO midgame evacuations
# Config 4: V16 with V14 target shortlist + NO midgame evacuations + NO staging attacks (should behave exactly like V15/V14 in 2-player)

def run_matchup(v16_agent, v14_agent, num_games=20):
    v16_wins = 0
    v14_wins = 0
    for i in range(num_games):
        swap = (i % 2 == 1)
        seed = 3000 + i
        env = make("orbit_wars", debug=False, configuration={"seed": seed})
        agents = [v14_agent, v16_agent] if swap else [v16_agent, v14_agent]
        result = env.run(agents)
        final = result[-1]
        r0, r1 = final[0]['reward'], final[1]['reward']
        if r0 is None or r1 is None:
            continue
        v16_reward = r1 if swap else r0
        v14_reward = r0 if swap else r1
        if v16_reward > v14_reward:
            v16_wins += 1
        elif v16_reward < v14_reward:
            v14_wins += 1
    return v16_wins

def load_agents(temp_v14_dir):
    # Load V14
    sys.path.insert(0, temp_v14_dir)
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
    v14_module = importlib.import_module("main")
    v14_agent = v14_module.agent
    sys.path.remove(temp_v14_dir)
    
    # Load V16 (current buddy_model)
    sys.path.insert(0, os.path.abspath("buddy_model"))
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
    v16_module = importlib.import_module("main")
    v16_agent = v16_module.agent
    sys.path.remove(os.path.abspath("buddy_model"))
    
    return v16_agent, v14_agent

def patch_file(filepath, target, replacement):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if target not in content:
        print(f"Warning: target not found in {filepath}")
    patched = content.replace(target, replacement)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched)

def main():
    v14_tar = "submission_v14.tar.gz"
    temp_v14_dir = os.path.abspath("temp_ablation_v14")
    os.makedirs(temp_v14_dir, exist_ok=True)
    with tarfile.open(v14_tar) as tar:
        tar.extractall(temp_v14_dir)
        
    planner_path = os.path.join("buddy_model", "orbit_lite", "planner_core.py")
    main_path = os.path.join("buddy_model", "main.py")
    
    # Backup files
    shutil.copy2(planner_path, planner_path + ".bak")
    shutil.copy2(main_path, main_path + ".bak")
    
    try:
        # --- Config 1: Current V16 ---
        v16, v14 = load_agents(temp_v14_dir)
        w1 = run_matchup(v16, v14, 10)
        print(f"Config 1 (Full V16) Win rate vs V14: {w1/10*100:.1f}% ({w1}/10 wins)")
        
        # --- Config 2: Revert Target Shortlist to proximity ---
        # We replace the attack_pref scoring in planner_core.py
        target_pref = """    attack_pref = torch.where(
        attack_mask,
        -proximity + 2.0 * prod - 20.0 * blocked_from_all_sources.float(),
        torch.full_like(proximity, float("-inf"))
    )"""
        replacement_pref = """    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float("-inf")))"""
        patch_file(planner_path, target_pref, replacement_pref)
        
        v16, v14 = load_agents(temp_v14_dir)
        w2 = run_matchup(v16, v14, 10)
        print(f"Config 2 (V16 + V14 shortlist) Win rate vs V14: {w2/10*100:.1f}% ({w2}/10 wins)")
        
        # --- Config 3: V16 + V14 shortlist + NO midgame evacuations ---
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
        patch_file(main_path, target_evac, replacement_evac)
        
        v16, v14 = load_agents(temp_v14_dir)
        w3 = run_matchup(v16, v14, 10)
        print(f"Config 3 (V16 + V14 shortlist + NO midgame evac) Win rate vs V14: {w3/10*100:.1f}% ({w3}/10 wins)")
        
        # --- Config 4: V16 + V14 shortlist + NO midgame evac + NO staging ---
        # To disable staging, we just don't run the redirect_mask block or redirect_mask = False
        target_redirect = "active = active | redirect_mask"
        replacement_redirect = "active = active # | redirect_mask"
        patch_file(main_path, target_redirect, replacement_redirect)
        
        v16, v14 = load_agents(temp_v14_dir)
        w4 = run_matchup(v16, v14, 10)
        print(f"Config 4 (V16 + V14 shortlist + NO evac + NO staging) Win rate vs V14: {w4/10*100:.1f}% ({w4}/10 wins)")
        
    finally:
        # Restore backups
        shutil.copy2(planner_path + ".bak", planner_path)
        shutil.copy2(main_path + ".bak", main_path)
        os.remove(planner_path + ".bak")
        os.remove(main_path + ".bak")
        shutil.rmtree(temp_v14_dir)

if __name__ == "__main__":
    main()
