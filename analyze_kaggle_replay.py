import json, sys

with open('replay.html', 'r', encoding='utf-8', errors='ignore') as f:
    html = f.read()

start_str = 'window.kaggle = '
si = html.find(start_str) + len(start_str)
ei = html.find('</script>', si)
data, _ = json.JSONDecoder().raw_decode(html[si:ei].strip().rstrip(';'))
steps = data.get('environment', {}).get('steps', [])
print(f'Total turns: {len(steps)}')

final = steps[-1]
r0 = final[0].get('reward', 0)
r1 = final[1].get('reward', 0) if len(final) > 1 else 0
winner = 'P0' if r0 > r1 else ('P1' if r1 > r0 else 'DRAW')
print(f'Final rewards: P0={r0}  P1={r1}  -> Winner: {winner}')

# per-turn tracking
print()
print('Turn | P0_ships | P1_ships | P0_planets | P1_planets | P0_fleets | P1_fleets')
print('-'*80)

chain_start = None
wasted_attacks = []
prev_p0_planets = set()
prev_p1_planets = set()

for idx, step in enumerate(steps):
    obs = step[0].get('observation', {})
    planets = obs.get('planets', [])
    fleets  = obs.get('fleets', []) or []

    p0_planet_ids = set(int(p[0]) for p in planets if int(p[1]) == 0)
    p1_planet_ids = set(int(p[0]) for p in planets if int(p[1]) == 1)
    p0s = sum(float(p[5]) for p in planets if int(p[1]) == 0)
    p1s = sum(float(p[5]) for p in planets if int(p[1]) == 1)
    p0f = sum(float(f[6]) for f in fleets if f and len(f) > 6 and int(f[1]) == 0)
    p1f = sum(float(f[6]) for f in fleets if f and len(f) > 6 and int(f[1]) == 1)

    if idx % 50 == 0 or idx == len(steps)-1:
        print(f'{idx:4d} | {p0s:8.0f} | {p1s:8.0f} | {len(p0_planet_ids):10d} | {len(p1_planet_ids):10d} | {p0f:9.0f} | {p1f:9.0f}')

    # Track P0 planet losses (chain-loss detection)
    lost_by_p0 = prev_p0_planets - p0_planet_ids
    gained_by_p0 = p0_planet_ids - prev_p0_planets
    if lost_by_p0 and idx > 10:
        if chain_start is None:
            chain_start = idx
    else:
        if chain_start is not None and idx - chain_start >= 3:
            print(f'  !! Chain-loss: turns {chain_start}-{idx-1} (P0 lost {idx-chain_start} turns in a row)')
        chain_start = None

    prev_p0_planets = p0_planet_ids
    prev_p1_planets = p1_planet_ids

print()
print('=== WASTED ATTACK DETECTION ===')
capture_turns = {}
prev_mine = set()
for idx, step in enumerate(steps):
    obs = step[0].get('observation', {})
    planets = obs.get('planets', [])
    mine = set(int(p[0]) for p in planets if int(p[1]) == 0)
    gained = mine - prev_mine
    lost = prev_mine - mine
    for pg in gained:
        capture_turns[pg] = idx
    for pl in lost:
        if pl in capture_turns and (idx - capture_turns[pl]) <= 20:
            print(f'  Planet {pl}: captured turn {capture_turns[pl]}, lost turn {idx} (held {idx-capture_turns[pl]} turns)')
    prev_mine = mine

print()
print('=== IDLE SHIP ACCUMULATION ===')
for idx, step in enumerate(steps):
    obs = step[0].get('observation', {})
    planets = obs.get('planets', [])
    fleets  = obs.get('fleets', []) or []
    for p in planets:
        if int(p[1]) == 0 and float(p[5]) > 300:
            my_fleets = [f for f in fleets if f and len(f) > 6 and int(f[1]) == 0]
            if len(my_fleets) < 2:
                print(f'  Turn {idx}: Planet {int(p[0])} has {float(p[5]):.0f} idle ships, only {len(my_fleets)} fleets in air')
