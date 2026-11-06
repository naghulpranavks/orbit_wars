import os
import sys
import tarfile
import shutil
import importlib
from kaggle_environments import make

# We will extract V14 as our base, and test adding each V15 feature.
# We will run 30 games for each configuration vs V14.
# Configuration 1: Pure V14 vs V14 (Control, should be ~50%)
# Configuration 2: V14 + Neutral Blitz ROI Boost
# Configuration 3: V14 + Endgame Evacuation (step >= 480)
# Configuration 4: V14 + Regroup Sun-Rerouting (Stepping Stones)
# Configuration 5: V14 + ETA-Aware Reinforcement
# Configuration 6: V14 + Coordinated Regrouping

def extract_clean_v14(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open("submission_v14.tar.gz") as tar:
        tar.extractall(target_dir)

def load_agent(agent_dir):
    sys.path.insert(0, agent_dir)
    # Clear sys.modules
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key:
            sys.modules.pop(key, None)
    module = importlib.import_module("main")
    agent_func = module.agent
    sys.path.remove(agent_dir)
    return agent_func

def run_matchup(agent_test, agent_v14, num_games=30):
    test_wins = 0
    v14_wins = 0
    draws = 0
    for i in range(num_games):
        swap = (i % 2 == 1)
        seed = 5000 + i
        env = make("orbit_wars", debug=False, configuration={"seed": seed})
        agents = [agent_v14, agent_test] if swap else [agent_test, agent_v14]
        result = env.run(agents)
        final = result[-1]
        r0, r1 = final[0]['reward'], final[1]['reward']
        if r0 is None or r1 is None:
            continue
        test_reward = r1 if swap else r0
        v14_reward = r0 if swap else r1
        if test_reward > v14_reward:
            test_wins += 1
        elif test_reward < v14_reward:
            v14_wins += 1
        else:
            draws += 1
    return test_wins, v14_wins, draws

def patch_file(filepath, target, replacement):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if target not in content:
        raise ValueError(f"Target not found in {filepath}: {target[:100]}")
    patched = content.replace(target, replacement)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched)

def main():
    v14_ref_dir = os.path.abspath("v14_ref")
    v14_test_dir = os.path.abspath("v14_test")
    
    # Extract baseline V14 (reference)
    extract_clean_v14(v14_ref_dir)
    agent_v14 = load_agent(v14_ref_dir)
    
    # -------------------------------------------------------------
    # Config 1: Control (V14 vs V14)
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    agent_test = load_agent(v14_test_dir)
    w1, l1, d1 = run_matchup(agent_test, agent_v14, 20)
    print(f"Config 1 (Control V14): Wins={w1}, Losses={l1}, Draws={d1} (Win rate: {w1/20*100:.1f}%)")
    
    # -------------------------------------------------------------
    # Config 2: V14 + Neutral Blitz ROI Boost
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    main_py = os.path.join(v14_test_dir, "main.py")
    # Patch main.py to add the ROI boost
    target_score = '    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)'
    replacement_score = """    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)
    step = int(obs.step.item())
    if step < 80:
        is_neutral_target = obs.is_neutral[cand_tgt_slot]
        score = torch.where(is_neutral_target, score + 0.3, score)"""
    patch_file(main_py, target_score, replacement_score)
    
    agent_test = load_agent(v14_test_dir)
    w2, l2, d2 = run_matchup(agent_test, agent_v14, 30)
    print(f"Config 2 (V14 + Neutral Blitz): Wins={w2}, Losses={l2}, Draws={d2} (Win rate: {w2/30*100:.1f}%)")
    
    # -------------------------------------------------------------
    # Config 3: V14 + Endgame Evacuation (step >= 480)
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    # Copy plan_endgame_evacuations from buddy_model/main.py
    # We will just write a patch that adds the function and calls it in run_turn
    main_py = os.path.join(v14_test_dir, "main.py")
    
    # Let's read the function plan_endgame_evacuations from buddy_model/main.py
    with open(os.path.join("buddy_model", "main.py"), "r", encoding="utf-8") as f:
        v16_main = f.read()
    
    # Extract the function plan_endgame_evacuations
    start_func = v16_main.find("def plan_endgame_evacuations")
    end_func = v16_main.find("def plan_midgame_evacuations")
    plan_evac_code = v16_main[start_func:end_func]
    
    # Insert it before plan_lite_waves in main.py
    with open(main_py, "r", encoding="utf-8") as f:
        v14_main_content = f.read()
    idx = v14_main_content.find("def plan_lite_waves")
    v14_main_content = v14_main_content[:idx] + plan_evac_code + "\n\n" + v14_main_content[idx:]
    with open(main_py, "w", encoding="utf-8") as f:
        f.write(v14_main_content)
        
    # Patch run_turn to call it
    target_run = """    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    entries = disambiguate_duplicate_launches(entries)"""
    
    replacement_run = """    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    
    dtype = obs.ships.dtype
    launched = torch.zeros(obs.P, dtype=dtype, device=device)
    launched.scatter_add_(0, entries.source_slots, torch.where(entries.valid, entries.ships, torch.zeros_like(entries.ships)))
    leftover = (obs.ships.to(dtype) - launched).clamp(min=0.0)
    
    if step >= 480:
        evac_entries = plan_endgame_evacuations(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=status,
            leftover=leftover,
            config=config
        )
        entries = concat_launch_entries([entries, evac_entries])
        
    entries = disambiguate_duplicate_launches(entries)"""
    patch_file(main_py, target_run, replacement_run)
    
    agent_test = load_agent(v14_test_dir)
    w3, l3, d3 = run_matchup(agent_test, agent_v14, 30)
    print(f"Config 3 (V14 + Endgame Evac): Wins={w3}, Losses={l3}, Draws={d3} (Win rate: {w3/30*100:.1f}%)")
    
    # Cleanup reference and test dirs at the end
    shutil.rmtree(v14_ref_dir, ignore_errors=True)
    shutil.rmtree(v14_test_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
