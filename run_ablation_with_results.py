import os
import sys
import tarfile
import shutil
import importlib
from kaggle_environments import make

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

def patch_file(filepath, target, replacement):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if target not in content:
        raise ValueError(f"Target not found in {filepath}: {target[:100]}")
    patched = content.replace(target, replacement)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched)

def run_matchup(agent_test, agent_v14, config_name, num_games=30):
    test_wins = 0
    v14_wins = 0
    draws = 0
    print(f"\n--- Starting Matchup for {config_name} ---", flush=True)
    for i in range(num_games):
        swap = (i % 2 == 1)
        seed = 6000 + i
        env = make("orbit_wars", debug=False, configuration={"seed": seed})
        agents = [agent_v14, agent_test] if swap else [agent_test, agent_v14]
        result = env.run(agents)
        final = result[-1]
        r0, r1 = final[0]['reward'], final[1]['reward']
        if r0 is None or r1 is None:
            print(f"Game {i} (seed {seed}): None reward", flush=True)
            continue
        test_reward = r1 if swap else r0
        v14_reward = r0 if swap else r1
        if test_reward > v14_reward:
            test_wins += 1
            outcome = "Test Win"
        elif test_reward < v14_reward:
            v14_wins += 1
            outcome = "V14 Win"
        else:
            draws += 1
            outcome = "Draw"
        print(f"Game {i} (seed {seed}, swap={swap}): {outcome} (Test={test_reward}, V14={v14_reward})", flush=True)
    win_rate = test_wins / num_games * 100
    print(f"=== {config_name} Results: Wins={test_wins}, Losses={v14_wins}, Draws={draws} (Win rate: {win_rate:.1f}%) ===", flush=True)
    return test_wins, v14_wins, draws

def main():
    v14_ref_dir = os.path.abspath("v14_ref")
    v14_test_dir = os.path.abspath("v14_test")
    
    extract_clean_v14(v14_ref_dir)
    agent_v14 = load_agent(v14_ref_dir)
    
    with open(os.path.join("buddy_model", "main.py"), "r", encoding="utf-8") as f:
        v16_main = f.read()
    with open(os.path.join("buddy_model", "orbit_lite", "planner_core.py"), "r", encoding="utf-8") as f:
        v16_planner = f.read()

    start_sun = v16_planner.find("def build_sun_blocked_matrix")
    end_sun = v16_planner.find("def _plan_regroup")
    sun_code = v16_planner[start_sun:end_sun]
        
    # -------------------------------------------------------------
    # Config 4: V14 + Regroup Sun-Rerouting (Stepping Stones)
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    main_py = os.path.join(v14_test_dir, "main.py")
    planner_py = os.path.join(v14_test_dir, "orbit_lite", "planner_core.py")
    
    with open(planner_py, "r", encoding="utf-8") as f:
        planner_content = f.read()
    idx = planner_content.find("def _plan_regroup")
    planner_content = planner_content[:idx] + sun_code + "\n\n" + planner_content[idx:]
    with open(planner_py, "w", encoding="utf-8") as f:
        f.write(planner_content)
        
    patch_file(main_py, "score_candidates,", "score_candidates,\n    build_sun_blocked_matrix,")
    
    target_active = """    regroup_active = reachable_mask(
        movement, source_idx=src_idx, target_idx=dst_idx,
        fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1),
        eta_cap=torch.full((T,), float(config.max_regroup_time), device=device),
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        src_idx.unsqueeze(1),                                                    # [S, 1]
        dst_idx.unsqueeze(0),                                                     # [1, T]
        regroup_cap.unsqueeze(1),                                                 # [S, 1]
        active=regroup_active,
    )"""
    
    replacement_active = """    regroup_active = reachable_mask(
        movement, source_idx=src_idx, target_idx=dst_idx,
        fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1),
        eta_cap=torch.full((T,), float(config.max_regroup_time), device=device),
    ).squeeze(-1)                                                                # [S, T]

    sun_blocked = build_sun_blocked_matrix(obs, center=50.0, radius=10.0)
    blocked_st = sun_blocked[src_idx.clamp(0, P - 1).view(S, 1), dst_idx.clamp(0, P - 1).view(1, T)]

    friendly_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        friendly_mask = friendly_mask & ~comet
    friendly_idx = torch.where(friendly_mask)[0]
    B_friendly = friendly_idx.shape[0]

    if B_friendly > 0:
        ok_s_to_b = ~sun_blocked[src_idx.clamp(0, P - 1).view(S, 1), friendly_idx.view(1, B_friendly)]
        ok_b_to_t = ~sun_blocked[friendly_idx.view(B_friendly, 1), dst_idx.clamp(0, P - 1).view(1, T)]
        valid_stone = ok_s_to_b.unsqueeze(1) & ok_b_to_t.T.unsqueeze(0)

        x0 = movement.x[0]
        y0 = movement.y[0]
        src_x = x0[src_idx.clamp(0, P - 1)].view(S, 1)
        src_y = y0[src_idx.clamp(0, P - 1)].view(S, 1)
        friendly_x = x0[friendly_idx].view(1, B_friendly)
        friendly_y = y0[friendly_idx].view(1, B_friendly)
        dst_x = x0[dst_idx.clamp(0, P - 1)].view(T, 1)
        dst_y = y0[dst_idx.clamp(0, P - 1)].view(T, 1)

        d_sb = torch.sqrt((src_x - friendly_x)**2 + (src_y - friendly_y)**2)
        d_bt = torch.sqrt((friendly_x.T - dst_x.T)**2 + (friendly_y.T - dst_y.T)**2)
        total_dist = d_sb.unsqueeze(1) + d_bt.T.unsqueeze(0)
        total_dist = torch.where(valid_stone, total_dist, torch.full_like(total_dist, float("inf")))

        min_val, best_b_idx = total_dist.min(dim=2)
        stone_planet_idx = friendly_idx[best_b_idx]

        redirect_mask = blocked_st & torch.isfinite(min_val)
        dst_idx_m = torch.where(redirect_mask, stone_planet_idx, dst_idx.view(1, T).expand(S, T))
        regroup_active = regroup_active | redirect_mask
    else:
        dst_idx_m = dst_idx.view(1, T).expand(S, T)

    aim = intercept_angle(
        movement,
        src_idx.unsqueeze(1),                                                    # [S, 1]
        dst_idx_m,                                                               # [S, T]
        regroup_cap.unsqueeze(1),                                                 # [S, 1]
        active=regroup_active,
    )"""
    patch_file(planner_py, target_active, replacement_active)
    
    target_owner = """    dst_owner = owner[dst_idx.clamp(0, P - 1)]                                  # [T, H+1]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)             # [S, T]
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)"""
    
    replacement_owner = """    dst_owner = owner[dst_idx_m.clamp(0, P - 1)]                                 # [S, T, H+1]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)             # [S, T]
    owner_at_k = dst_owner.gather(-1, k.unsqueeze(-1)).squeeze(-1)"""
    patch_file(planner_py, target_owner, replacement_owner)
    
    target_neq = """    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)"""
    replacement_neq = """    src_neq_dst = src_idx.view(S, 1) != dst_idx_m"""
    patch_file(planner_py, target_neq, replacement_neq)
    
    target_best = """    best_dst = dst_idx[best_t]                                                   # [S]"""
    replacement_best = """    best_dst = dst_idx_m[torch.arange(S, device=device), best_t]                 # [S]"""
    patch_file(planner_py, target_best, replacement_best)
    
    agent_test = load_agent(v14_test_dir)
    run_matchup(agent_test, agent_v14, "Config 4 (V14 + Regroup Sun-Rerouting)", 30)

    # -------------------------------------------------------------
    # Config 5: V14 + ETA-Aware Reinforcement
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    main_py = os.path.join(v14_test_dir, "main.py")
    planner_py = os.path.join(v14_test_dir, "orbit_lite", "planner_core.py")
    
    start_reinf = v16_main.find("def build_reinforcement_tensor")
    end_reinf = v16_main.find("def build_threat_weights")
    reinf_code = v16_main[start_reinf:end_reinf]
    
    with open(main_py, "r", encoding="utf-8") as f:
        v14_main_content = f.read()
    idx = v14_main_content.find("def plan_lite_waves")
    v14_main_content = v14_main_content[:idx] + reinf_code + "\n\n" + v14_main_content[idx:]
    with open(main_py, "w", encoding="utf-8") as f:
        f.write(v14_main_content)
        
    target_floor = """    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)"""
    replacement_floor = """    reinforcement = build_reinforcement_tensor(obs, cache, target_idx, K_eta, beta=0.18)
    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid, reinforcement=reinforcement)"""
    patch_file(main_py, target_floor, replacement_floor)
    
    start_cf = v16_planner.find("def capture_floor")
    end_cf = v16_planner.find("def attack_target_mask")
    cf_code = v16_planner[start_cf:end_cf]
    
    with open(planner_py, "r", encoding="utf-8") as f:
        p_content = f.read()
    start_idx = p_content.find("def capture_floor")
    end_idx = p_content.find("def attack_target_mask")
    p_content = p_content[:start_idx] + cf_code + "\n\n" + p_content[end_idx:]
    with open(planner_py, "w", encoding="utf-8") as f:
        f.write(p_content)
        
    agent_test = load_agent(v14_test_dir)
    run_matchup(agent_test, agent_v14, "Config 5 (V14 + ETA-Aware Reinforcement)", 30)

    # -------------------------------------------------------------
    # Config 6: V14 + Coordinated Regrouping
    # -------------------------------------------------------------
    extract_clean_v14(v14_test_dir)
    planner_py = os.path.join(v14_test_dir, "orbit_lite", "planner_core.py")
    
    start_regroup = v16_planner.find("def _plan_regroup")
    end_regroup = v16_planner.find("def _empty_entries")
    regroup_code = v16_planner[start_regroup:end_regroup]
    
    with open(planner_py, "r", encoding="utf-8") as f:
        p_content = f.read()
    idx = p_content.find("def _plan_regroup")
    p_content = p_content[:idx] + sun_code + "\n\n" + p_content[idx:]
    
    start_idx = p_content.find("def _plan_regroup")
    end_idx = p_content.find("def _empty_entries")
    p_content = p_content[:start_idx] + regroup_code + "\n\n" + p_content[end_idx:]
    with open(planner_py, "w", encoding="utf-8") as f:
        f.write(p_content)
        
    agent_test = load_agent(v14_test_dir)
    run_matchup(agent_test, agent_v14, "Config 6 (V14 + Coordinated Regrouping)", 30)
    
    shutil.rmtree(v14_ref_dir, ignore_errors=True)
    shutil.rmtree(v14_test_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
