import sys, os, json
sys.path.insert(0, 'buddy_model')
from kaggle_environments import make

v14 = os.path.join('buddy_model', 'main.py')
v7 = os.path.join('v7_bot', 'main.py')

losses = []
wins = 0
for g in range(10):
    env = make('orbit_wars', debug=True)
    result = env.run([v14, v7])
    final = result[-1]
    r0, r1 = final[0]['reward'], final[1]['reward']
    status = 'W' if r0 > r1 else 'L' if r0 < r1 else 'D'
    print(f'Game {g+1}: [{status}] V14={r0:.0f} V7={r1:.0f}')
    if status == 'W':
        wins += 1
    if status == 'L':
        replay = []
        for step_idx, step in enumerate(result):
            if step_idx % 50 == 0 or step_idx == len(result)-1:
                obs = step[0].get('observation', {})
                planets = obs.get('planets', [])
                my_count = sum(1 for p in planets if int(p[1]) == 0)
                enemy_count = sum(1 for p in planets if int(p[1]) == 1)
                my_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0)
                enemy_ships = sum(float(p[5]) for p in planets if int(p[1]) == 1)
                replay.append({
                    'step': step_idx,
                    'my_planets': my_count,
                    'enemy_planets': enemy_count,
                    'my_ships': int(my_ships),
                    'enemy_ships': int(enemy_ships)
                })
        losses.append({'game': g+1, 'timeline': replay})

print()
print(f'Record: {wins}W / {10-wins}L')
if losses:
    print()
    print(f'LOSS ANALYSIS ({len(losses)} losses):')
    for loss in losses:
        gnum = loss['game']
        print(f'  Game {gnum}:')
        for snap in loss['timeline']:
            s = snap['step']
            mp = snap['my_planets']
            ep = snap['enemy_planets']
            ms = snap['my_ships']
            es = snap['enemy_ships']
            print(f'    Step {s:3d}: planets {mp}v{ep} | ships {ms}v{es}')
else:
    print('NO LOSSES - V14 won all 10 games!')
