import sys, importlib, os
from kaggle_environments import make
from concurrent.futures import ProcessPoolExecutor

def run_game(seed):
    # Load V23
    V23_DIR = os.path.abspath('buddy_model')
    sys.path.insert(0, V23_DIR)
    for k in list(sys.modules):
        if 'orbit_lite' in k or k == 'main': sys.modules.pop(k, None)
    m_v23 = importlib.import_module('main')
    v23 = m_v23.agent
    sys.path.remove(V23_DIR)

    # Load V22
    V22_DIR = os.path.abspath('v22_bot')
    sys.path.insert(0, V22_DIR)
    for k in list(sys.modules):
        if 'orbit_lite' in k or k == 'main': sys.modules.pop(k, None)
    m_v22 = importlib.import_module('main')
    v22 = m_v22.agent
    sys.path.remove(V22_DIR)

    # Load V15
    V15_DIR = os.path.abspath('v15_bot')
    sys.path.insert(0, V15_DIR)
    for k in list(sys.modules):
        if 'orbit_lite' in k or k == 'main': sys.modules.pop(k, None)
    m_v15 = importlib.import_module('main')
    v15 = m_v15.agent
    sys.path.remove(V15_DIR)

    # Load V14
    V14_DIR = os.path.abspath('v14_bot')
    sys.path.insert(0, V14_DIR)
    for k in list(sys.modules):
        if 'orbit_lite' in k or k == 'main': sys.modules.pop(k, None)
    m_v14 = importlib.import_module('main')
    v14 = m_v14.agent
    sys.path.remove(V14_DIR)

    env = make('orbit_wars', debug=False, configuration={'seed': seed})
    result = env.run([v23, v22, v15, v14])
    rewards = [a.reward for a in env.steps[-1]]
    if rewards[0] is None:
        return 'ERROR'
    if rewards[0] > max(rewards[1:]):
        return 'WIN'
    elif rewards[0] == max(rewards):
        return 'DRAW'
    else:
        return 'LOSS'

if __name__ == '__main__':
    seeds = list(range(3000, 3015))
    results = {'WIN': 0, 'LOSS': 0, 'DRAW': 0, 'ERROR': 0}
    with ProcessPoolExecutor(max_workers=5) as executor:
        for seed, res in zip(seeds, executor.map(run_game, seeds)):
            print(f'Seed {seed}: {res}', flush=True)
            results[res] += 1
    print('Final results:', results)
