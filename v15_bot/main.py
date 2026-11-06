
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
# V14: V7's PROVEN DEFAULTS (no risky parameter changes)
# ============================================================
@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 22
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
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
    return contrib.sum(dim=0)


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


def plan_lite_waves(*, movement, obs, obs_tensors, cache, garrison_status, prod, alive_by_step, config, player_count):
    P = obs.P; device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1]); H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H)); W = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()): return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(obs, obs_tensors, garrison_status, cache, config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask)
    if not bool(target_exists.any()): return _empty_entries(device, dtype)
    S = int(source_idx.shape[0]); T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(garrison_status, source_idx=source_idx, source_ships=source_ships, H_eff=H_eff, player_id=pid)
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    reinforcement = build_reinforcement_tensor(obs, cache, target_idx, K_eta, beta=0.18)
    floor = capture_floor(garrison_status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid, reinforcement=reinforcement)
    K = int(floor.shape[-1])
    sizes = drain.view(S, 1).expand(S, T).floor()
    active = reachable_mask(movement, source_idx=source_idx, target_idx=target_idx, fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)
    aim = intercept_angle(movement, source_idx.unsqueeze(1), target_idx.unsqueeze(0), sizes, active=active)
    angle = aim["angle"]; eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = viable & clears_floor & (sizes >= 1.0) & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    L = 1; C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L); cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]
    launches = make_launch_set(source_slots=cand_src, target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L), ships=cand_send, eta=cand_eta, valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = score_candidates(garrison_status, prod=prod, alive_by_step=alive_by_step, player_count=int(player_count), launches=launches, player_id=pid)
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    step = int(obs.step.item())
    if step < 80:
        is_neutral_target = obs.is_neutral[cand_tgt_slot]
        score = torch.where(is_neutral_target, score + 0.3, score)
    wave_entries, leftover = _greedy_select(P=P, W=W, device=device, dtype=dtype, score=score, cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta, cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(), target_exists=target_exists, roi_threshold=float(config.roi_threshold))
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
    entries = plan_lite_waves(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, garrison_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step, config=config, player_count=int(player_count))
    
    # Endgame Evacuation Protocol
    if step >= 480:
        dtype = obs.ships.dtype
        launched = torch.zeros(obs.P, dtype=dtype, device=device)
        launched.scatter_add_(0, entries.source_slots, torch.where(entries.valid, entries.ships, torch.zeros_like(entries.ships)))
        leftover = (obs.ships.to(dtype) - launched).clamp(min=0.0)
        
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

    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id))
    apply_private_planned_launches(movement=movement, launches=launches, owner_id=int(obs.player_id), obs_tensors=obs_tensors)
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


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
    """V14: Clean V7 engine + anti-snipe + score awareness. No Citadel Veto. No Ghost Fleet."""
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
