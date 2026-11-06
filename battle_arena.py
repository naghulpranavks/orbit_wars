"""
V14 Real Battle Arena
Tests V14 against REAL opponents:
1. V7 (our previous best, scored 781 on leaderboard)
2. An Elite Bot (smart multi-wave attacker with defense)
"""
import sys
import os
import time

from kaggle_environments import make

# A smart bot that balances offense/defense with proper fleet sizing
ELITE_BOT = """
import math

def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    
    my_planets = [p for p in planets if int(p[1]) == player_id]
    enemy_planets = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player_id]
    neutral_planets = [p for p in planets if int(p[1]) == -1]
    targets = enemy_planets + neutral_planets
    
    if not my_planets or not targets:
        return []
    
    # Analyze incoming threats
    threat_to = {}
    for f in fleets:
        if int(f[0]) < 0 or int(f[1]) == player_id:
            continue
        fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), float(f[6])
        dx, dy = math.cos(fa), math.sin(fa)
        for p in my_planets:
            px, py = float(p[2]), float(p[3])
            vx, vy = px - fx, py - fy
            dot = vx*dx + vy*dy
            cross = abs(vx*dy - vy*dx)
            if dot > 0 and cross < float(p[4]) + 2.0:
                pid = int(p[0])
                threat_to[pid] = threat_to.get(pid, 0) + fs
    
    moves = []
    used = {}
    
    # Phase 1: Defense - reinforce threatened planets
    for p in my_planets:
        pid = int(p[0])
        threat = threat_to.get(pid, 0)
        garrison = float(p[5])
        if threat > garrison * 0.8:
            # Find reinforcements
            for rp in sorted(my_planets, key=lambda x: (float(x[2])-float(p[2]))**2 + (float(x[3])-float(p[3]))**2):
                rpid = int(rp[0])
                if rpid == pid:
                    continue
                avail = float(rp[5]) - used.get(rpid, 0) - 10
                if avail < 5:
                    continue
                send = min(avail, threat - garrison + 10)
                if send >= 5:
                    dx = float(p[2]) - float(rp[2])
                    dy = float(p[3]) - float(rp[3])
                    moves.append([rpid, math.atan2(dy, dx), int(send)])
                    used[rpid] = used.get(rpid, 0) + send
                    break
    
    # Phase 2: Attack - concentrate on best target
    # Score targets by production / (defense * distance)
    best_targets = []
    for tgt in targets:
        tgt_ships = float(tgt[5])
        tgt_prod = float(tgt[6])
        
        # Find nearest source
        min_dist = float('inf')
        for src in my_planets:
            dx = float(tgt[2]) - float(src[2])
            dy = float(tgt[3]) - float(src[3])
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
        
        owner = int(tgt[1])
        if owner >= 0 and owner != player_id:
            value = tgt_prod * 2  # Enemy: double value
        else:
            value = tgt_prod
        
        score = value / (tgt_ships + 1) / max(1, min_dist * 0.1)
        best_targets.append((score, tgt))
    
    best_targets.sort(key=lambda x: -x[0])
    
    # Attack top 3 targets with concentrated force
    for _, tgt in best_targets[:3]:
        tgt_ships = float(tgt[5])
        tgt_prod = float(tgt[6])
        needed = tgt_ships + tgt_prod * 10 + 10
        
        # Gather force from multiple sources
        remaining = needed
        sources = sorted(my_planets, key=lambda s: (float(s[2])-float(tgt[2]))**2 + (float(s[3])-float(tgt[3]))**2)
        
        for src in sources:
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid, 0) - max(10, float(src[5]) * 0.2)
            if avail < 5 or remaining <= 0:
                continue
            
            send = min(avail, remaining)
            if send >= 5:
                dx = float(tgt[2]) - float(src[2])
                dy = float(tgt[3]) - float(src[3])
                moves.append([sid, math.atan2(dy, dx), int(send)])
                used[sid] = used.get(sid, 0) + send
                remaining -= send
    
    return moves
"""


def run_match(env, agent1_path, agent2_path_or_code, name):
    env.reset()
    
    if os.path.exists(str(agent2_path_or_code)):
        a2 = agent2_path_or_code
        temp = None
    else:
        temp = os.path.join(os.path.dirname(__file__), '_temp_opp.py')
        with open(temp, 'w') as f:
            f.write(agent2_path_or_code)
        a2 = temp
    
    try:
        result = env.run([agent1_path, a2])
        final = result[-1]
        r0 = final[0]['reward']
        r1 = final[1]['reward']
        s0 = final[0]['status']
        
        if s0 == 'ERROR':
            return 'ERROR', r0, r1
        elif r0 > r1:
            return 'WIN', r0, r1
        elif r0 < r1:
            return 'LOSS', r0, r1
        else:
            return 'DRAW', r0, r1
    finally:
        if temp and os.path.exists(temp):
            os.remove(temp)


def main():
    print("=" * 60)
    print("V14 REAL BATTLE ARENA")
    print("=" * 60)
    
    v14_path = os.path.join(os.path.dirname(__file__), 'buddy_model', 'main.py')
    v7_path = os.path.join(os.path.dirname(__file__), 'v7_bot', 'main.py')
    
    opponents = [
        ("V7 (Leaderboard 781)", v7_path),
        ("Elite Bot", ELITE_BOT),
    ]
    
    total_w, total_l, total_e = 0, 0, 0
    
    for opp_name, opp in opponents:
        print(f"\n--- V14 vs {opp_name} ---")
        w, l, e = 0, 0, 0
        
        for g in range(5):
            env = make("orbit_wars", debug=True)
            start = time.time()
            result, s0, s1 = run_match(env, v14_path, opp, opp_name)
            elapsed = time.time() - start
            icon = {"WIN":"W","LOSS":"L","DRAW":"D","ERROR":"X"}[result]
            print(f"  Game {g+1}: [{icon}] V14={s0:.0f} vs {opp_name}={s1:.0f} ({elapsed:.1f}s)")
            
            if result == 'WIN': w += 1
            elif result == 'LOSS': l += 1
            else: e += 1
        
        total_w += w; total_l += l; total_e += e
        print(f"  Record: {w}W / {l}L / {e}E")
    
    total = total_w + total_l + total_e
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_w}W / {total_l}L / {total_e}E out of {total}")
    print(f"WIN RATE: {total_w/max(1,total)*100:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
