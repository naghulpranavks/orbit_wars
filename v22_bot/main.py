
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    LaunchEntries,
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_sun_blocked_matrix,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    is_comet_planet,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


# ============================================================
# V17: V15's PROVEN PARAMETERS restored (V16 Optuna over-fit to V14)
# ============================================================
@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 22
    max_sources_per_lane: int = 20
    max_offensive_targets: int = 20
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 10
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    planet_pressure = contrib.sum(dim=0)

    # Fleet-aware pressure: add enemy in-transit fleets based on their current position.
    # This lets the regroup respond to incoming attacks before they arrive on planet.
    fleet_pressure = torch.zeros(P, dtype=dtype, device=device)
    if int(obs.F) > 0:
        f_enemy = obs.f_alive & (obs.f_owner >= 0) & (obs.f_owner != float(player_id))
        if f_enemy.any():
            f_x = obs.f_x[f_enemy].to(dtype)      # [Fe]
            f_y = obs.f_y[f_enemy].to(dtype)      # [Fe]
            f_s = obs.f_ships[f_enemy].to(dtype)  # [Fe]
            px = obs.x.to(dtype)                   # [P]
            py = obs.y.to(dtype)                   # [P]
            # Distance from each planet to each enemy fleet position [P, Fe]
            dist_pf = torch.sqrt(
                (px.view(P, 1) - f_x.view(1, -1)) ** 2 +
                (py.view(P, 1) - f_y.view(1, -1)) ** 2
            ).clamp(min=1e-6)
            f_speed = fleet_speed(f_s.clamp(min=1e-6))           # [Fe]
            f_reach = (f_speed * float(horizon)).view(1, -1)      # [1, Fe]
            f_decay = (1.0 - dist_pf / f_reach).clamp(min=0.0)   # [P, Fe]
            alive_p = obs.alive.to(dtype).view(P, 1)              # only live planets
            fleet_pressure = (f_s.view(1, -1) * f_decay * alive_p).sum(dim=1)

    return planet_pressure + fleet_pressure


def build_reinforcement_tensor(obs, cache, target_idx, k_max, beta=0.18):
    device = obs.device
    dtype = obs.ships.dtype
    P = obs.P
    T = target_idx.shape[0]
    K = int(k_max)

    if K <= 0 or T == 0:
        return torch.zeros((T, K), dtype=dtype, device=device)

    enemy_mask = obs.is_enemy & obs.alive
    if not enemy_mask.any():
        return torch.zeros((T, K), dtype=dtype, device=device)

    ships_e = torch.where(enemy_mask, obs.ships, torch.zeros_like(obs.ships))
    speed_e = fleet_speed(ships_e.clamp(min=1e-6))

    dist_et = cache.cross_dist[0][:, target_idx]

    eta_et = dist_et / speed_e.view(P, 1).clamp(min=1e-6)

    k_grid = torch.arange(1, K + 1, dtype=dtype, device=device).view(1, 1, K)
    eta_et_3d = eta_et.unsqueeze(-1)

    risk_factor = ((k_grid - eta_et_3d) / 8.0).clamp(0.0, 1.0)

    ships_e_3d = ships_e.view(P, 1, 1)
    risk_contrib = ships_e_3d * risk_factor * float(beta)
    risk_tk = risk_contrib.sum(dim=0)

    return risk_tk


def build_threat_weights(obs, player_count) -> Tensor:
    """Calculate threat weights for each player in the game."""
    device = obs.device
    dtype = obs.ships.dtype
    A = int(player_count)
    owner = obs.owner_abs.long()
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)
    player_ships = torch.zeros(A, dtype=dtype, device=device)
    player_prod = torch.zeros(A, dtype=dtype, device=device)
    valid = obs.alive & (owner >= 0) & (owner < A)
    if valid.any():
        player_ships.scatter_add_(0, owner[valid], ships[valid])
        player_prod.scatter_add_(0, owner[valid], prod[valid])
    f_valid = obs.f_alive & (obs.f_owner >= 0) & (obs.f_owner < A)
    if f_valid.any():
        f_owner = obs.f_owner[f_valid].long()
        f_ships = obs.f_ships[f_valid].to(dtype)
        player_ships.scatter_add_(0, f_owner, f_ships)
    threat = player_prod + 0.02 * player_ships
    return threat.clamp(min=1e-3)


def plan_endgame_evacuations(*, movement, obs, obs_tensors, cache, garrison_status, leftover, config):
    P = obs.P; device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1]); H = max(H_axis - 1, 0)

    # 1. Identify doomed planets: owned by me now, but owner at the final turn of the horizon (index H) is not me.
    owner_at_500 = garrison_status.owner[:, H]
    doomed_mask = obs.owned & obs.alive & (owner_at_500 != pid) & (leftover >= float(config.min_ships_to_launch))

    if not doomed_mask.any():
        return _empty_entries(device, dtype)

    doomed_idx = torch.where(doomed_mask)[0]
    S = doomed_idx.shape[0]

    # 2. Identify candidate destination planets (friendly bases): currently owned by me
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet

    # We want to route to a safe friendly base if possible (owner at step 500 is still me)
    safe_dst_mask = dst_mask & (owner_at_500 == pid)
    if not safe_dst_mask.any():
        # Fall back to any friendly base
        safe_dst_mask = dst_mask

    if not safe_dst_mask.any():
        return _empty_entries(device, dtype)

    dst_idx = torch.where(safe_dst_mask)[0]
    T = dst_idx.shape[0]

    # We evaluate paths from all doomed sources to all destinations
    src_idx_2d = doomed_idx.unsqueeze(1)
    dst_idx_2d = dst_idx.unsqueeze(0)
    fleet_sizes_2d = leftover[doomed_idx].unsqueeze(1)

    aim = intercept_angle(
        movement,
        src_idx_2d,
        dst_idx_2d,
        fleet_sizes_2d,
    )
    viable = aim["viable"]
    angle = aim["angle"]
    eta = aim["eta"]

    src_neq_dst = doomed_idx.view(S, 1) != dst_idx.view(1, T)
    valid_paths = viable & src_neq_dst

    if not valid_paths.any():
        return _empty_entries(device, dtype)

    dist_st = cache.cross_dist[0][doomed_idx][:, dst_idx]
    dist_st = torch.where(valid_paths, dist_st, torch.full_like(dist_st, float("inf")))

    min_dist, best_t_idx = dist_st.min(dim=1)
    valid_evac = torch.isfinite(min_dist)

    evac_src = doomed_idx
    evac_dst = dst_idx[best_t_idx]

    t_idx_unsqueezed = best_t_idx.unsqueeze(1)
    evac_angle = angle.gather(1, t_idx_unsqueezed).squeeze(1)
    evac_eta = eta.gather(1, t_idx_unsqueezed).squeeze(1)
    evac_ships = leftover[doomed_idx].floor()

    valid_evac = valid_evac & (evac_ships >= float(config.min_ships_to_launch))

    return LaunchEntries(
        source_slots=evac_src,
        target_slots=evac_dst,
        ships=torch.where(valid_evac, evac_ships, torch.zeros_like(evac_ships)),
        angle=torch.where(valid_evac, evac_angle, torch.zeros_like(evac_angle)),
        eta=torch.where(valid_evac, evac_eta, torch.ones_like(evac_eta)),
        valid=valid_evac,
    )


def plan_midgame_evacuations(*, movement, obs, obs_tensors, cache, garrison_status, leftover, config):
    P = obs.P; device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1]); H = max(H_axis - 1, 0)
    if H <= 0:
        return _empty_entries(device, dtype)
    owner_h = garrison_status.owner[:, 1:]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)
    any_flip = flips.any(dim=-1)
    doomed_mask = any_flip & obs.alive & (leftover >= float(config.min_ships_to_launch))
    if not doomed_mask.any():
        return _empty_entries(device, dtype)
    doomed_idx = torch.where(doomed_mask)[0]
    S = doomed_idx.shape[0]
    flips_float = flips[doomed_idx].float()
    k_flip = flips_float.argmax(dim=-1) + 1
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    k_flip_clamp = k_flip.clamp(max=H)
    owner_at_flip = garrison_status.owner.T[k_flip_clamp]
    safe_dst = dst_mask.view(1, P) & (owner_at_flip == pid)
    safe_dst = safe_dst & (torch.arange(P, device=device).view(1, P) != doomed_idx.view(S, 1))
    src_idx_2d = doomed_idx.unsqueeze(1)
    dst_idx_2d = torch.arange(P, device=device).unsqueeze(0)
    fleet_sizes_2d = leftover[doomed_idx].unsqueeze(1)
    aim = intercept_angle(
        movement,
        src_idx_2d,
        dst_idx_2d,
        fleet_sizes_2d,
    )
    viable = aim["viable"]
    angle = aim["angle"]
    eta = aim["eta"]
    valid_paths = safe_dst & viable
    if not valid_paths.any():
        return _empty_entries(device, dtype)
    x0 = movement.x[0]
    y0 = movement.y[0]
    src_x = x0[doomed_idx].view(S, 1)
    src_y = y0[doomed_idx].view(S, 1)
    dst_x = x0.view(1, P)
    dst_y = y0.view(1, P)
    dist_sp = torch.sqrt((src_x - dst_x)**2 + (src_y - dst_y)**2)
    dist_sp = torch.where(valid_paths, dist_sp, torch.full_like(dist_sp, float("inf")))
    min_dist, best_p_idx = dist_sp.min(dim=1)
    valid_evac = torch.isfinite(min_dist)
    if not valid_evac.any():
        return _empty_entries(device, dtype)
    evac_src = doomed_idx
    evac_dst = best_p_idx
    t_idx_unsqueezed = best_p_idx.unsqueeze(1)
    evac_angle = angle.gather(1, t_idx_unsqueezed).squeeze(1)
    evac_eta = eta.gather(1, t_idx_unsqueezed).squeeze(1)
    evac_ships = leftover[doomed_idx].floor()
    valid_evac = valid_evac & (evac_ships >= float(config.min_ships_to_launch))
    return LaunchEntries(
        source_slots=evac_src,
        target_slots=evac_dst,
        ships=torch.where(valid_evac, evac_ships, torch.zeros_like(evac_ships)),
        angle=torch.where(valid_evac, evac_angle, torch.zeros_like(evac_angle)),
        eta=torch.where(valid_evac, evac_eta, torch.ones_like(evac_eta)),
        valid=valid_evac,
    )


def plan_lite_waves(*, movement, obs, obs_tensors, cache, garrison_status, prod, alive_by_step, config, player_count, threat_weights=None):
    P = obs.P; device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1]); H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H)); W = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()): return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)

    # Use short-horizon status for target shortlisting and safe_drain.
    # (garrison_status_long is available for future use but safe_drain with
    # H_eff=45 paralysed the bot by returning near-zero drain on every planet.)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask
    )
    print('Target exists:', target_exists.sum().item())
    print('Source mask:', source_mask.sum().item())
    if not bool(target_exists.any()): return _empty_entries(device, dtype)
    S = int(source_idx.shape[0]); T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(garrison_status, source_idx=source_idx, source_ships=source_ships, H_eff=H_eff, player_id=pid)

    x0 = movement.x[0]
    y0 = movement.y[0]
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    reinforcement = build_reinforcement_tensor(obs, cache, target_idx, K_eta, beta=0.18)
    
    # Revert to standard capture overhead so we don't cripple expansion
    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)
    K = int(floor.shape[-1])

    sizes = drain.view(S, 1).expand(S, T).floor()
    active = reachable_mask(movement, source_idx=source_idx, target_idx=target_idx, fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)

    # Attack Sun-Rerouting (disabled: regressed performance in ablation)
    sun_blocked = build_sun_blocked_matrix(obs, center=50.0, radius=10.0)
    blocked_st = sun_blocked[source_idx.clamp(0, P - 1).view(S, 1), target_idx.clamp(0, P - 1).view(1, T)]
    friendly_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        friendly_mask = friendly_mask & ~comet
    friendly_idx = torch.where(friendly_mask)[0]
    B_friendly = 0 # friendly_idx.shape[0] — disabled

    if B_friendly > 0:
        ok_s_to_b = ~sun_blocked[source_idx.clamp(0, P - 1).view(S, 1), friendly_idx.view(1, B_friendly)]
        ok_b_to_t = ~sun_blocked[friendly_idx.view(B_friendly, 1), target_idx.clamp(0, P - 1).view(1, T)]
        valid_stone = ok_s_to_b.unsqueeze(1) & ok_b_to_t.T.unsqueeze(0)

        friendly_x = x0[friendly_idx].view(1, B_friendly)
        friendly_y = y0[friendly_idx].view(1, B_friendly)
        src_x = x0[source_idx.clamp(0, P - 1)].view(S, 1)
        src_y = y0[source_idx.clamp(0, P - 1)].view(S, 1)
        dst_x = x0[target_idx.clamp(0, P - 1)].view(T, 1)
        dst_y = y0[target_idx.clamp(0, P - 1)].view(T, 1)

        d_sb = torch.sqrt((src_x - friendly_x)**2 + (src_y - friendly_y)**2)
        d_bt = torch.sqrt((friendly_x.T - dst_x.T)**2 + (friendly_y.T - dst_y.T)**2)
        total_dist = d_sb.unsqueeze(1) + d_bt.T.unsqueeze(0)
        total_dist = torch.where(valid_stone, total_dist, torch.full_like(total_dist, float("inf")))

        min_val, best_b_idx = total_dist.min(dim=2)
        stone_planet_idx = friendly_idx[best_b_idx]

        redirect_mask = blocked_st & torch.isfinite(min_val)
        target_idx_m = torch.where(redirect_mask, stone_planet_idx, target_idx.view(1, T).expand(S, T))
        active = active | redirect_mask
    else:
        redirect_mask = torch.zeros((S, T), dtype=torch.bool, device=device)
        target_idx_m = target_idx.view(1, T).expand(S, T)

    aim = intercept_angle(movement, source_idx.unsqueeze(1), target_idx_m, sizes, active=active)
    angle = aim["angle"]; eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    src_neq_tgt = source_idx.view(S, 1) != target_idx_m
    valid = viable & clears_floor & (sizes >= 1.0) & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    # --- V21 Combinatorial Expansion Reverted (L=1) ---
    L = 1
    C1 = S * T
    src1 = source_idx.view(S, 1).expand(S, T).reshape(C1)
    tgt_m1 = target_idx_m.reshape(C1)
    tgt_orig1 = target_idx.view(1, T).expand(S, T).reshape(C1)
    tgt_short1 = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C1)
    
    send1 = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C1)
    ang1 = angle.reshape(C1)
    eta1 = torch.where(valid, eta, torch.ones_like(eta)).reshape(C1)
    act1 = valid.reshape(C1)
    
    cand_src_1 = src1.unsqueeze(-1)
    cand_send_1 = send1.unsqueeze(-1)
    cand_angle_1 = ang1.unsqueeze(-1)
    cand_eta_1 = eta1.unsqueeze(-1)
    cand_active_1 = act1.unsqueeze(-1)
    cand_tgt_slot_1 = tgt_m1
    cand_tgt_orig_1 = tgt_orig1
    cand_tgt_short_1 = tgt_short1
    cand_valid_1 = act1
    
    cand_src = cand_src_1
    cand_send = cand_send_1
    cand_angle = cand_angle_1
    cand_eta = cand_eta_1
    cand_active = cand_active_1
    cand_tgt_slot = cand_tgt_slot_1
    cand_tgt_orig = cand_tgt_orig_1
    cand_tgt_short = cand_tgt_short_1
    cand_valid = cand_valid_1
        
    C = cand_src.shape[0]
    cand_is_def = target_is_mine[cand_tgt_short]
    launches = make_launch_set(source_slots=cand_src, target_slots=cand_tgt_orig.unsqueeze(-1).expand(C, L), ships=cand_send, eta=cand_eta, valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid, threat_weights=threat_weights)
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    step = int(obs.step.item())
    if step < 80:
        is_neutral_target = obs.is_neutral[cand_tgt_orig.clamp(0, P - 1)]
        score = torch.where(is_neutral_target, score + 0.3, score)

    # Staging attack score penalty (for extra transit time — only fires if B_friendly > 0)
    if B_friendly > 0:
        b_x = x0[stone_planet_idx]
        b_y = y0[stone_planet_idx]
        t_x = x0[target_idx].view(1, T)
        t_y = y0[target_idx].view(1, T)
        dist_bt = torch.sqrt((b_x - t_x)**2 + (b_y - t_y)**2)
        cand_speed = fleet_speed(cand_send.reshape(S, T).clamp(min=1.0))
        eta_bt = dist_bt / cand_speed.clamp(min=1e-6)
        score = torch.where(redirect_mask.reshape(C), score - 0.05 * eta_bt.reshape(C), score)

    wave_entries, leftover = _greedy_select(P=P, W=W, device=device, dtype=dtype, score=score, cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta, cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_orig=cand_tgt_orig, cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(), target_exists=target_exists, roi_threshold=float(config.roi_threshold))
    if not bool(config.enable_regroup): return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status, leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass, config=config, H=H)
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors, *, config, player_count, memory):
    device = obs_tensors["planets"].device; obs = parse_obs(obs_tensors)
    if obs.P == 0: return empty_action_row(device)
    
    # Endgame Horizon Tapering: Reduce search horizon as we approach turn 500
    step = int(obs_tensors["step"].item())
    remaining = max(0, 500 - step)
    dynamic_horizon = min(int(config.horizon), remaining)
    if dynamic_horizon <= 0:
        return empty_action_row(device)
    config = dataclasses.replace(config, horizon=dynamic_horizon)
    
    movement = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=_movement_config(config, player_count=int(player_count)), cached_movement=getattr(memory, "movement", None))
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon)); H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H); alive_by_step = movement.alive_by_step[: H + 1]
    # No threat_weights: V15 did not use them and scored higher on Kaggle.
    # build_threat_weights biases toward attacking the strongest enemy (harmful).
    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count), threat_weights=None)
    
    # Evacuation Protocols
    dtype = obs.ships.dtype
    launched = torch.zeros(obs.P, dtype=dtype, device=device)
    launched.scatter_add_(0, entries.source_slots, torch.where(entries.valid, entries.ships, torch.zeros_like(entries.ships)))
    leftover = (obs.ships.to(dtype) - launched).clamp(min=0.0)
    
    # 1. Apply primary entries to update movement cache
    entries = disambiguate_duplicate_launches(entries)
    launches_initial = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id))
    apply_private_planned_launches(movement=movement, launches=launches_initial, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    
    # 2. Get updated garrison status reflecting our reinforcements
    status_updated = movement.garrison_status(max_horizon=H)
    
    if step >= 480:
        evac_entries = plan_endgame_evacuations(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=status_updated,
            leftover=leftover,
            config=config
        )
    else:
        evac_entries = _empty_entries(device, dtype)
        
    evac_entries = disambiguate_duplicate_launches(evac_entries)
    launches_evac = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=movement, entries=evac_entries, player_id=int(obs.player_id))
    apply_private_planned_launches(movement=movement, launches=launches_evac, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    
    combined_entries = concat_launch_entries([entries, evac_entries])
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(combined_entries, planet_ids=planet_ids)


CONFIG_4P = dataclasses.replace(ProducerLiteConfig(), horizon=13, max_sources_per_lane=6, max_defensive_targets=2, max_regroup_time=6.0, max_regroup_targets_per_source=8)

def _config_for(player_count): return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()

class ProducerLiteMemory:
    def __init__(self): self.movement = None; self.cached_player_count = None; self.last_sparse_action_row = None
    def reset(self): self.movement = None; self.cached_player_count = None; self.last_sparse_action_row = None

class ProducerLiteRuntime:
    def __init__(self, memory=None): self.memory = memory if memory is not None else ProducerLiteMemory()
    def reset(self): self.memory.reset()
    def tensor_action(self, obs_tensors):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()): mem.cached_player_count = None
        if mem.cached_player_count is None: mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(obs_tensors, config=config, player_count=int(mem.cached_player_count), memory=mem)
        mem.last_sparse_action_row = row
        return row

_RUNTIME = ProducerLiteRuntime()


def agent(obs):
    """V17: V15 proven parameters + V16 infrastructure. No threat_weight bias."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step

    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)

    moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    
    # Return V7's moves directly — no hacks, no overrides.
    # V7's engine already handles:
    # - Exact orbital intercept aiming (intercept_aim.py)
    # - Fleet path verification (swept-pair collision)
    # - Sun collision avoidance (built into the engine)
    # - Garrison management (safe_drain)
    # - Target prioritization (score_candidates)
    # - Defensive reinforcement (regroup)
    return moves
