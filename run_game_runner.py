import sys
import os
import importlib
from kaggle_environments import make

def load_agent(directory):
    sys.path.insert(0, directory)
    for key in list(sys.modules.keys()):
        if "orbit_lite" in key or "main" in key or "agent" in key:
            sys.modules.pop(key, None)
    
    try:
        module = importlib.import_module("main")
    except ModuleNotFoundError:
        module = importlib.import_module("agent")
        
    loaded_agent = getattr(module, "agent", None)
    if loaded_agent is None:
        loaded_agent = getattr(module, "act", None)
    
    sys.path.remove(directory)
    return loaded_agent

import time
def time_agent(agent_fn):
    def wrapper(obs, conf):
        t0 = time.time()
        res = agent_fn(obs, conf)
        dt = time.time() - t0
        if dt > 0.5: print(f'Turn took {dt:.3f}s')
        return res
    return wrapper
def main():
    if len(sys.argv) < 5:
        print("ERROR: Missing args")
        return
    seed = int(sys.argv[1])
    test_dir = sys.argv[2]
    ref_dir = sys.argv[3]
    swap = sys.argv[4].lower() == 'true'
    
    ref_agent = load_agent(ref_dir)
    test_agent = time_agent(load_agent(test_dir))
    
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    agents = [ref_agent, test_agent] if swap else [test_agent, ref_agent]
    
    try:
        res = env.run(agents)
        rewards = res[-1][0]["reward"], res[-1][1]["reward"]
        if 'error' in res[-1][0]: print('Error P1:', res[-1][0]['error'])
        if 'error' in res[-1][1]: print('Error P2:', res[-1][1]['error'])
        if rewards[0] == rewards[1]:
            print("RESULT:DRAW")
        elif swap:
            if rewards[1] > rewards[0]: print("RESULT:TEST_WIN")
            else: print("RESULT:REF_WIN")
        else:
            if rewards[0] > rewards[1]: print("RESULT:TEST_WIN")
            else: print("RESULT:REF_WIN")
    except Exception as e:
        print("ERROR", e)

if __name__ == "__main__":
    main()
