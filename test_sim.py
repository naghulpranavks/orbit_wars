from kaggle_environments import make
import sys
sys.path.insert(0, 'buddy_model')
import main as main_agent

env = make('orbit_wars', debug=True, configuration={'seed': 3000})
state = env.reset(2)[0]
print('Starting simulation...')
for step in range(100):
    obs = state.observation
    actions = main_agent.agent(obs)
    state = env.step([actions, {}])[0]
    if len(actions) > 0:
        print(f'Turn {step}: Launched {len(actions)} fleets. Actions: {actions}')
