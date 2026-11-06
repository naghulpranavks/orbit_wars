import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

MAX_SHIP_SPEED = 2.0
LOG_1000 = math.log(1000.0)

# ==========================================
# 1. THE PHYSICS ENGINE
# ==========================================

def get_fleet_speed(ships):
    if ships < 1.0: ships = 1.0
    ratio = min(math.log(ships) / LOG_1000, 1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * (ratio ** 1.5)

def ray_intersects_circle(fx, fy, angle, px, py, radius):
    vx = px - fx
    vy = py - fy
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    t = vx * dx + vy * dy
    if t < 0:
        return False, float('inf') 
        
    cx = fx + t * dx
    cy = fy + t * dy
    
    dist_sq = (cx - px)**2 + (cy - py)**2
    if dist_sq > radius**2:
        return False, float('inf') 
        
    dt = math.sqrt(radius**2 - dist_sq)
    return True, t - dt

def resolve_fleet_targets(raw_fleets, planets):
    parsed_fleets = []
    for f in raw_fleets:
        fid, owner, fx, fy, angle, _, ships = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
        speed = get_fleet_speed(ships)
        
        best_target = None
        min_dist = float('inf')
        
        for p in planets:
            hits, dist = ray_intersects_circle(fx, fy, angle, p["x"], p["y"], p["radius"])
            if hits and dist < min_dist:
                min_dist = dist
                best_target = p["id"]
                
        eta = math.ceil(min_dist / speed) if best_target is not None else 999
        parsed_fleets.append({"id": fid, "owner": owner, "ships": ships, "target_id": best_target, "eta": eta})
    return parsed_fleets

# ==========================================
# 2. V6 PSYCHOLOGICAL ORACLE LOGIC
# ==========================================

def agent(obs):
    my_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    
    planets = []
    for p_data in raw_planets:
        p = Planet(*p_data)
        planets.append({"id": p.id, "owner": p.owner, "x": p.x, "y": p.y, "radius": p.radius, "ships": p.ships, "production": p.production})
        
    parsed_fleets = resolve_fleet_targets(raw_fleets, planets)
    
    friendly_arrivals = defaultdict(lambda: defaultdict(int))
    enemy_arrivals = defaultdict(lambda: defaultdict(int))
    
    for f in parsed_fleets:
        if f["target_id"] is not None:
            if f["owner"] == my_id:
                friendly_arrivals[f["target_id"]][f["eta"]] += f["ships"]
            else:
                enemy_arrivals[f["target_id"]][f["eta"]] += f["ships"]
                
    my_planets = [p for p in planets if p["owner"] == my_id]
    enemy_planets = [p for p in planets if p["owner"] not in [-1, my_id]]
    
    # Failsafe 2: Total Reserve Tracking
    total_enemy_ships = sum(p["ships"] for p in enemy_planets) + sum(f["ships"] for f in parsed_fleets if f["owner"] not in [-1, my_id])
    total_my_ships = sum(p["ships"] for p in my_planets) + sum(f["ships"] for f in parsed_fleets if f["owner"] == my_id)
    
    actions = []
    
    # Dynamic Game Phases
    is_early_game = step < 100
    is_late_game = step > 350
    reserve_threshold = 5 if is_late_game else 15
    
    ghost_launched = False
    
    for src in my_planets:
        best_target = None
        best_roi = -1.0
        ships_to_send = 0
        
        # --- EVACUATION & CITADEL PROTOCOL ---
        garrison = src["ships"]
        is_doomed = False
        for turn in range(1, 6):
            garrison += src["production"]
            garrison += friendly_arrivals[src["id"]][turn]
            garrison -= enemy_arrivals[src["id"]][turn]
            if garrison < 0:
                is_doomed = True
                break
                
        if is_doomed:
            strategic_value = src["production"]
            # Citadel Protocol: Sacrifice useless border planets to counter-attack
            if strategic_value < 3 and total_my_ships > total_enemy_ships * 1.2:
                targets = sorted(enemy_planets, key=lambda x: x["production"], reverse=True)
                if targets:
                    best_target = targets[0]
                    ships_to_send = src["ships"]
            else:
                # Standard Evacuation: Flee to highest ROI target
                for tgt in planets:
                    if tgt["owner"] != my_id:
                        d = math.ceil(math.hypot(tgt["x"] - src["x"], tgt["y"] - src["y"]) / get_fleet_speed(src["ships"]))
                        if d == 0: d = 1
                        roi = (tgt["production"] + 0.1) / (d * 0.1)
                        if roi > best_roi:
                            best_roi, best_target, ships_to_send = roi, tgt, src["ships"]
        else:
            # --- OFFENSE, SNIPERS & GHOST FLEETS ---
            if src["ships"] < reserve_threshold:
                continue
                
            # Ghost Fleet Protocol: Launch 1 ship at enemy capital to trigger panic (every 10 turns)
            # Failsafe 1: Disabled in early game so we don't waste actions before profiling the enemy
            if not ghost_launched and step > 50 and step % 10 == 0 and enemy_planets and src["ships"] > 50:
                ghost_target = max(enemy_planets, key=lambda x: x["production"])
                dy = ghost_target["y"] - src["y"]
                dx = ghost_target["x"] - src["x"]
                angle = math.atan2(dy, dx)
                actions.append([src["id"], angle, 1])
                src["ships"] -= 1
                ghost_launched = True
            
            for tgt in planets:
                if src["id"] == tgt["id"]:
                    continue
                    
                speed = get_fleet_speed(src["ships"])
                d = math.ceil(math.hypot(tgt["x"] - src["x"], tgt["y"] - src["y"]) / speed)
                if d == 0: d = 1
                
                tgt_garrison = tgt["ships"]
                owner_at_impact = tgt["owner"]
                
                # Sniper Protocol: Simulate the exact turn the enemy captures a neutral planet
                for turn in range(1, d + 1):
                    if owner_at_impact != -1: tgt_garrison += tgt["production"]
                    tgt_garrison += friendly_arrivals[tgt["id"]][turn]
                    
                    if owner_at_impact == -1 and enemy_arrivals[tgt["id"]][turn] > 0:
                        tgt_garrison -= enemy_arrivals[tgt["id"]][turn]
                        if tgt_garrison < 0:
                            owner_at_impact = -2 # Enemy captured the neutral planet!
                            tgt_garrison = abs(tgt_garrison) # Surviving enemy ships
                    else:
                        tgt_garrison -= enemy_arrivals[tgt["id"]][turn]
                
                if owner_at_impact == my_id:
                    # Defense
                    if tgt_garrison < 0:
                        cost = abs(tgt_garrison) + 1
                        if src["ships"] > cost:
                            roi = 1000.0
                            if roi > best_roi:
                                best_roi, best_target, ships_to_send = roi, tgt, cost
                else:
                    # Capture
                    # Failsafe 3: The 10% Over-Calculation Margin
                    cost = tgt_garrison + 1
                    if cost < 0: cost = 1
                    cost = int(cost * 1.1) + 5
                    
                    if src["ships"] > cost:
                        phase_bonus = 2.0 if (is_early_game and tgt["owner"] == -1) else 1.0
                        roi = ((tgt["production"] + 0.1) * phase_bonus) / (cost + (d * 0.5))
                        if roi > best_roi:
                            best_roi, best_target, ships_to_send = roi, tgt, cost
                            
        if best_target is not None and ships_to_send > 0:
            dy = best_target["y"] - src["y"]
            dx = best_target["x"] - src["x"]
            angle = math.atan2(dy, dx)
            
            actions.append([src["id"], angle, ships_to_send])
            src["ships"] -= ships_to_send
            
            eta = math.ceil(math.hypot(best_target["x"] - src["x"], best_target["y"] - src["y"]) / get_fleet_speed(ships_to_send))
            friendly_arrivals[best_target["id"]][eta] += ships_to_send

    return actions
