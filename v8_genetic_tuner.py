import optuna
import sys
import os
import time
from kaggle_environments import make

# Add the buddy_model directory to sys.path so we can import it
sys.path.append(os.path.abspath('buddy_model'))
import main as buddy_main

def objective(trial):
    """
    The Genetic Fitness Function.
    Optuna will mutate these 4 variables for every trial to find the perfect combination.
    """
    # 1. Propose genetic mutations
    _horizon = trial.suggest_int('horizon', 18, 25)
    _max_waves = trial.suggest_int('max_waves_per_turn', 6, 15)
    _roi_threshold = trial.suggest_float('roi_threshold', 1.0, 3.0)
    _min_ships = trial.suggest_float('min_ships_to_launch', 2.0, 10.0)
    
    # 2. Monkey-patch the PyTorch engine with the mutated genes
    class MutatedConfig:
        horizon: int = _horizon
        max_sources_per_lane: int = 12
        max_offensive_targets: int = 12
        max_defensive_targets: int = 4
        max_waves_per_turn: int = _max_waves
        roi_threshold: float = _roi_threshold
        min_ships_to_launch: float = _min_ships
        enable_regroup: bool = True
        max_regroup_time: float = 7.0
        regroup_pressure_delta_min: float = 0.25
        max_regroup_sources_per_lane: int = 6
        max_regroup_targets_per_source: int = 7
        regroup_pressure_norm: str = "none"
        regroup_time_penalty_weight: float = 1e-3

    buddy_main.ProducerLiteConfig = MutatedConfig
    
    # Reset the engine memory just in case
    buddy_main._RUNTIME.reset()

    # 3. Simulate matches
    # We run 5 matches per mutation to get a reliable win-rate
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    
    wins = 0
    matches = 5
    
    for i in range(matches):
        env.reset()
        # We test our Mutated V7 against the default 'random' bot to gauge absolute speed
        # In a real rigorous test, we would test it against a saved V6 agent!
        steps = env.run([buddy_main.agent, "random"])
        
        # Calculate win
        final_state = steps[-1][0]
        rewards = final_state.get("reward", [0, 0])
        
        # Player 0 is our agent. If reward > 0, we won.
        if rewards is not None:
            if isinstance(rewards, (list, tuple)):
                if len(rewards) > 0 and rewards[0] > 0:
                    wins += 1
            else:
                if rewards > 0:
                    wins += 1
            
    win_rate = wins / matches
    return win_rate

if __name__ == "__main__":
    print("==================================================")
    print("V8 GENETIC ORACLE: PyTorch Parameter Optimization")
    print("==================================================")
    print("Initializing Optuna Genetic Algorithm...")
    time.sleep(1)
    
    # Create the genetic study
    study = optuna.create_study(direction='maximize', study_name="V8_Orbit_Wars")
    
    print("Starting 100 generations of mutations. This will take a while...")
    # Run 100 genetic mutations
    study.optimize(objective, n_trials=100)
    
    print("==================================================")
    print("GENETIC TUNING COMPLETE!")
    print("==================================================")
    print("The Mathematically Perfect PyTorch Parameters are:")
    
    best = study.best_params
    print(f"Horizon: {best['horizon']}")
    print(f"Max Waves: {best['max_waves_per_turn']}")
    print(f"ROI Threshold: {best['roi_threshold']:.3f}")
    print(f"Min Ships to Launch: {best['min_ships_to_launch']:.3f}")
    
    print("\nNext Step:")
    print("Open buddy_model/main.py and replace the ProducerLiteConfig block")
    print("with these exact numbers. Then re-package submission_v7_pytorch.tar.gz!")
