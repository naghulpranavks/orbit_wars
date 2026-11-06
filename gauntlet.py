"""
V14 GAUNTLET: 100+ games against 5 hard opponents.
Each opponent implements a distinct competitive strategy.
"""
import sys, os, time, math, random
sys.path.insert(0, 'buddy_model')
from kaggle_environments import make

# ============================================================
# BOT 1: PRODUCTION RUSH (grab everything early, snowball)
# ============================================================
BOT_RUSH = '''
import math

def agent(obs):
    pid = int(obs.get("player",0) if isinstance(obs,dict) else obs.player)
    step = int(obs.get("step",0) if isinstance(obs,dict) else obs.step)
    planets = obs.get("planets",[]) if isinstance(obs,dict) else obs.planets
    fleets = obs.get("fleets",[]) if isinstance(obs,dict) else obs.fleets
    
    my = [p for p in planets if int(p[1])==pid]
    targets = [p for p in planets if int(p[0])>=0 and int(p[1])!=pid]
    if not my or not targets: return []
    
    # Sort targets: neutrals first (easy grabs), then weakest enemy
    neutrals = [t for t in targets if int(t[1])==-1]
    enemies = [t for t in targets if int(t[1])>=0]
    neutrals.sort(key=lambda t: float(t[5]))
    enemies.sort(key=lambda t: float(t[5]))
    targets = neutrals + enemies
    
    moves = []
    used = {}
    
    for tgt in targets[:12]:  # Attack up to 12 targets per turn
        tgt_ships = float(tgt[5])
        needed = tgt_ships + 3  # Just enough to capture
        
        # Find nearest source with enough ships
        best = None; best_d = 1e9
        for src in my:
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - 3
            if avail < needed: continue
            dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
            d = math.sqrt(dx*dx+dy*dy)
            if d < best_d: best_d=d; best=src
        
        if best and best_d < 80:
            sid = int(best[0])
            send = int(min(needed + 2, float(best[5]) - used.get(sid,0) - 3))
            if send >= 3:
                dx = float(tgt[2])-float(best[2]); dy = float(tgt[3])-float(best[3])
                moves.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0) + send
    
    return moves
'''

# ============================================================
# BOT 2: TURTLE FORTRESS (massive defense, surgical strikes)
# ============================================================
BOT_TURTLE = '''
import math

def agent(obs):
    pid = int(obs.get("player",0) if isinstance(obs,dict) else obs.player)
    step = int(obs.get("step",0) if isinstance(obs,dict) else obs.step)
    planets = obs.get("planets",[]) if isinstance(obs,dict) else obs.planets
    fleets = obs.get("fleets",[]) if isinstance(obs,dict) else obs.fleets
    
    my = [p for p in planets if int(p[1])==pid]
    enemies = [p for p in planets if int(p[0])>=0 and int(p[1])>=0 and int(p[1])!=pid]
    neutrals = [p for p in planets if int(p[0])>=0 and int(p[1])==-1]
    if not my: return []
    
    moves = []
    used = {}
    total_ships = sum(float(p[5]) for p in my)
    
    # Phase 1: Reinforce - send ships to highest-production planets
    my_by_prod = sorted(my, key=lambda p: -float(p[6]))
    for i, dest in enumerate(my_by_prod[:3]):
        did = int(dest[0])
        for src in my:
            sid = int(src[0])
            if sid == did: continue
            if float(src[6]) >= float(dest[6]): continue  # Don't drain high-prod
            avail = float(src[5]) - used.get(sid,0) - 5
            if avail > 20:
                send = int(avail * 0.3)
                if send >= 5:
                    dx = float(dest[2])-float(src[2]); dy = float(dest[3])-float(src[3])
                    moves.append([sid, math.atan2(dy,dx), send])
                    used[sid] = used.get(sid,0) + send
    
    # Phase 2: Attack only when we have 3x enemy ships on a planet
    targets = neutrals + sorted(enemies, key=lambda p: float(p[5]))
    for tgt in targets[:5]:
        needed = float(tgt[5]) * 3 + 10
        
        # Gather from multiple sources
        gathered = 0
        attacks = []
        for src in sorted(my, key=lambda s: (float(s[2])-float(tgt[2]))**2+(float(s[3])-float(tgt[3]))**2):
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - max(15, float(src[5])*0.4)
            if avail < 5: continue
            send = min(int(avail), int(needed - gathered))
            if send >= 5:
                dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
                attacks.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0) + send
                gathered += send
            if gathered >= needed: break
        
        if gathered >= needed * 0.8:
            moves.extend(attacks)
    
    return moves
'''

# ============================================================
# BOT 3: SNIPER (cheap captures + harassment)
# ============================================================
BOT_SNIPER = '''
import math

def agent(obs):
    pid = int(obs.get("player",0) if isinstance(obs,dict) else obs.player)
    step = int(obs.get("step",0) if isinstance(obs,dict) else obs.step)
    planets = obs.get("planets",[]) if isinstance(obs,dict) else obs.planets
    
    my = [p for p in planets if int(p[1])==pid]
    others = [p for p in planets if int(p[0])>=0 and int(p[1])!=pid]
    if not my: return []
    
    moves = []
    used = {}
    
    # Find all weak/undefended planets and snipe them
    weak = [p for p in others if float(p[5]) < 15]
    weak.sort(key=lambda p: float(p[5]))  # Weakest first
    
    for tgt in weak:
        tgt_ships = float(tgt[5])
        send_needed = int(tgt_ships) + 2
        
        # Find closest source
        best = None; best_d = 1e9
        for src in my:
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - 5
            if avail < send_needed: continue
            dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
            d = math.sqrt(dx*dx+dy*dy)
            if d < best_d: best_d=d; best=src
        
        if best and best_d < 70:
            sid = int(best[0])
            dx = float(tgt[2])-float(best[2]); dy = float(tgt[3])-float(best[3])
            moves.append([sid, math.atan2(dy,dx), send_needed])
            used[sid] = used.get(sid,0) + send_needed
    
    # Also attack highest-prod enemy with concentrated force
    enemies = [p for p in planets if int(p[0])>=0 and int(p[1])>=0 and int(p[1])!=pid]
    if enemies:
        target = max(enemies, key=lambda p: float(p[6]))
        needed = float(target[5]) * 2 + 20
        gathered = 0
        for src in sorted(my, key=lambda s: -(float(s[5])-used.get(int(s[0]),0))):
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - 8
            if avail < 5 or gathered >= needed: continue
            send = min(int(avail*0.6), int(needed-gathered))
            if send >= 5:
                dx = float(target[2])-float(src[2]); dy = float(target[3])-float(src[3])
                moves.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0)+send
                gathered += send
    
    return moves
'''

# ============================================================
# BOT 4: ADAPTIVE (changes strategy based on game state)
# ============================================================
BOT_ADAPTIVE = '''
import math

def agent(obs):
    pid = int(obs.get("player",0) if isinstance(obs,dict) else obs.player)
    step = int(obs.get("step",0) if isinstance(obs,dict) else obs.step)
    planets = obs.get("planets",[]) if isinstance(obs,dict) else obs.planets
    fleets = obs.get("fleets",[]) if isinstance(obs,dict) else obs.fleets
    
    my = [p for p in planets if int(p[1])==pid]
    enemies = [p for p in planets if int(p[0])>=0 and int(p[1])>=0 and int(p[1])!=pid]
    neutrals = [p for p in planets if int(p[0])>=0 and int(p[1])==-1]
    if not my: return []
    
    my_ships = sum(float(p[5]) for p in my)
    my_prod = sum(float(p[6]) for p in my)
    en_ships = sum(float(p[5]) for p in enemies)
    en_prod = sum(float(p[6]) for p in enemies)
    
    moves = []
    used = {}
    
    # EARLY GAME: Rush neutrals
    if step < 80:
        targets = sorted(neutrals, key=lambda t: float(t[5]))
        for tgt in targets[:10]:
            needed = int(float(tgt[5])) + 3
            best = None; best_d = 1e9
            for src in my:
                sid = int(src[0])
                avail = float(src[5]) - used.get(sid,0) - 2
                if avail < needed: continue
                dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
                d = math.sqrt(dx*dx+dy*dy)
                if d < best_d: best_d=d; best=src
            if best:
                sid = int(best[0])
                dx = float(tgt[2])-float(best[2]); dy = float(tgt[3])-float(best[3])
                moves.append([sid, math.atan2(dy,dx), needed])
                used[sid] = used.get(sid,0) + needed
        return moves
    
    # MID/LATE GAME: Adapt based on position
    winning = my_prod > en_prod * 1.2
    
    if winning:
        # Play safe: only attack weak targets
        targets = sorted(neutrals + [e for e in enemies if float(e[5])<30], key=lambda t: float(t[5]))
    else:
        # Aggressive: target enemy production
        targets = sorted(enemies, key=lambda t: -float(t[6])) + neutrals
    
    for tgt in targets[:8]:
        tgt_ships = float(tgt[5])
        needed = tgt_ships * (1.5 if int(tgt[1])>=0 else 1.0) + 5
        
        gathered = 0; attacks = []
        for src in sorted(my, key=lambda s: (float(s[2])-float(tgt[2]))**2+(float(s[3])-float(tgt[3]))**2):
            sid = int(src[0])
            keep = max(8, float(src[5]) * (0.3 if winning else 0.15))
            avail = float(src[5]) - used.get(sid,0) - keep
            if avail < 5 or gathered >= needed: continue
            send = min(int(avail), int(needed-gathered))
            if send >= 3:
                dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
                attacks.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0)+send
                gathered += send
        
        if gathered >= needed * 0.7:
            moves.extend(attacks)
    
    return moves
'''

# ============================================================
# BOT 5: CONCENTRATED HAMMER (picks ONE target, overwhelms it)
# ============================================================
BOT_HAMMER = '''
import math

def agent(obs):
    pid = int(obs.get("player",0) if isinstance(obs,dict) else obs.player)
    step = int(obs.get("step",0) if isinstance(obs,dict) else obs.step)
    planets = obs.get("planets",[]) if isinstance(obs,dict) else obs.planets
    
    my = [p for p in planets if int(p[1])==pid]
    targets = [p for p in planets if int(p[0])>=0 and int(p[1])!=pid]
    if not my or not targets: return []
    
    # Pick THE BEST single target: highest production / (defense + 1)
    def score(t):
        prod = float(t[6])
        defense = float(t[5])
        # Find min distance from any of our planets
        min_d = min(math.sqrt((float(t[2])-float(s[2]))**2+(float(t[3])-float(s[3]))**2) for s in my)
        owner = int(t[1])
        mult = 2.0 if owner >= 0 else 1.0
        return prod * mult / (defense + 1) / max(1, min_d * 0.05)
    
    targets.sort(key=score, reverse=True)
    
    moves = []
    used = {}
    
    # Attack top 2 targets with everything we can spare
    for tgt in targets[:2]:
        tgt_ships = float(tgt[5])
        needed = tgt_ships * 2.5 + 15
        
        gathered = 0
        for src in sorted(my, key=lambda s: (float(s[2])-float(tgt[2]))**2+(float(s[3])-float(tgt[3]))**2):
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - 5
            if avail < 5 or gathered >= needed: continue
            send = min(int(avail * 0.7), int(needed - gathered))
            if send >= 3:
                dx = float(tgt[2])-float(src[2]); dy = float(tgt[3])-float(src[3])
                moves.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0)+send
                gathered += send
    
    # Also grab any free neutrals nearby
    neutrals = [p for p in planets if int(p[0])>=0 and int(p[1])==-1 and float(p[5])<5]
    for n in neutrals[:5]:
        for src in my:
            sid = int(src[0])
            avail = float(src[5]) - used.get(sid,0) - 3
            if avail < int(float(n[5]))+2: continue
            dx = float(n[2])-float(src[2]); dy = float(n[3])-float(src[3])
            d = math.sqrt(dx*dx+dy*dy)
            if d < 40:
                send = int(float(n[5])) + 2
                moves.append([sid, math.atan2(dy,dx), send])
                used[sid] = used.get(sid,0)+send
                break
    
    return moves
'''


def run_match(v14_path, opp_code):
    env = make("orbit_wars", debug=True)
    env.reset()
    tmp = os.path.join(os.path.dirname(__file__), '_tmp_opp.py')
    with open(tmp, 'w') as f: f.write(opp_code)
    try:
        result = env.run([v14_path, tmp])
        final = result[-1]
        r0, r1 = final[0]['reward'], final[1]['reward']
        s0 = final[0]['status']
        if s0 == 'ERROR': return 'E', r0, r1
        if r0 > r1: return 'W', r0, r1
        if r0 < r1: return 'L', r0, r1
        return 'D', r0, r1
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def main():
    v14 = os.path.join(os.path.dirname(__file__), 'buddy_model', 'main.py')
    v7 = os.path.join(os.path.dirname(__file__), 'v7_bot', 'main.py')
    
    opponents = [
        ("Rush Bot", BOT_RUSH, 20),
        ("Turtle Bot", BOT_TURTLE, 20),
        ("Sniper Bot", BOT_SNIPER, 20),
        ("Adaptive Bot", BOT_ADAPTIVE, 20),
        ("Hammer Bot", BOT_HAMMER, 20),
    ]
    
    print("=" * 65)
    print("V14 GAUNTLET: 100 Games Against 5 Hard Opponents")
    print("=" * 65)
    
    grand_w, grand_l, grand_e = 0, 0, 0
    
    for name, code, n_games in opponents:
        print(f"\n--- V14 vs {name} ({n_games} games) ---")
        w, l, e = 0, 0, 0
        for g in range(n_games):
            start = time.time()
            result, s0, s1 = run_match(v14, code)
            elapsed = time.time() - start
            icon = {"W":"W","L":"L","D":"D","E":"X"}[result]
            print(f"  G{g+1:2d}: [{icon}] ({elapsed:.1f}s)", end="")
            if (g+1) % 5 == 0: print()
            if result == 'W': w += 1
            elif result == 'L': l += 1
            else: e += 1
        
        if n_games % 5 != 0: print()
        wr = w/max(1,n_games)*100
        print(f"  => {w}W / {l}L / {e}E  ({wr:.0f}%)")
        grand_w += w; grand_l += l; grand_e += e
    
    total = grand_w + grand_l + grand_e
    print(f"\n{'='*65}")
    print(f"GRAND TOTAL: {grand_w}W / {grand_l}L / {grand_e}E out of {total}")
    print(f"OVERALL WIN RATE: {grand_w/max(1,total)*100:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
