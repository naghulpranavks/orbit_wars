import os
import sys
import shutil
import tarfile
import subprocess
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed

# We will disable Optuna's verbose logging to keep the console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def extract_clean_v14(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open("submission_v14.tar.gz") as tar:
        tar.extractall(target_dir)

def run_game_subprocess(seed, test_dir, ref_dir, swap):
    cmd = [
        sys.executable,
        "run_game_runner.py",
        str(seed),
        test_dir,
        ref_dir,
        str(swap)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    stdout = res.stdout.strip()
    for line in stdout.split("\n"):
        if line.startswith("RESULT:"):
            parts = line.split(":")
            return parts[1]
    return "ERROR"

def objective(trial):
    # Propose mutated parameters
    horizon = trial.suggest_int("horizon", 18, 26)
    max_waves = trial.suggest_int("max_waves", 6, 14)
    roi_threshold = trial.suggest_float("roi_threshold", 1.0, 2.2)
    min_ships = trial.suggest_float("min_ships", 2.0, 8.0)
    max_regroup_time = trial.suggest_float("max_regroup_time", 4.0, 10.0)
    regroup_pressure_delta = trial.suggest_float("regroup_pressure_delta", 0.1, 0.5)
    
    test_dir = os.path.abspath("v16_tune_test")
    ref_dir = os.path.abspath("v14_ref")
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    shutil.copytree("buddy_model", test_dir)
    
    # Write the mutated config to v16_tune_test/main.py
    main_py = os.path.join(test_dir, "main.py")
    with open(main_py, "r", encoding="utf-8") as f:
        content = f.read()
        
    config_code = f"""@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = {horizon}
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = {max_waves}
    roi_threshold: float = {roi_threshold:.4f}
    min_ships_to_launch: float = {min_ships:.4f}
    enable_regroup: bool = True
    max_regroup_time: float = {max_regroup_time:.4f}
    regroup_pressure_delta_min: float = {regroup_pressure_delta:.4f}
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3"""

    # Find the start and end of ProducerLiteConfig in main.py
    start_idx = content.find("@dataclass(frozen=True)\nclass ProducerLiteConfig:")
    end_idx = content.find("def _movement_config")
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find ProducerLiteConfig block in main.py")
        
    mutated_content = content[:start_idx] + config_code + "\n\n\n" + content[end_idx:]
    with open(main_py, "w", encoding="utf-8") as f:
        f.write(mutated_content)
        
    # Run 12 games in parallel to evaluate
    num_games = 12
    wins = 0
    futures = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(num_games):
            swap = (i % 2 == 1)
            seed = 8000 + i
            futures.append(executor.submit(run_game_subprocess, seed, test_dir, ref_dir, swap))
            
        for fut in as_completed(futures):
            outcome = fut.result()
            if outcome == "TEST_WIN":
                wins += 1
                
    shutil.rmtree(test_dir, ignore_errors=True)
    win_rate = wins / num_games
    print(f"Trial {trial.number}: Win Rate = {win_rate*100:.1f}% (H={horizon}, W={max_waves}, ROI={roi_threshold:.2f}, Min={min_ships:.2f}, RegroupTime={max_regroup_time:.2f})", flush=True)
    return win_rate

def main():
    v14_ref_dir = os.path.abspath("v14_ref")
    extract_clean_v14(v14_ref_dir)
    
    # We must write run_game_runner.py back to make sure it's available
    # (caller handles this but we do a sanity check)
    if not os.path.exists("run_game_runner.py"):
        print("Error: run_game_runner.py is missing!")
        sys.exit(1)
        
    print("=== Launching Parallel Hyperparameter Optimization ===", flush=True)
    print("Tuning V16 parameters against V14 baseline across 35 trials...", flush=True)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=35)
    
    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)
    best = study.best_params
    print(f"Best Win Rate: {study.best_value*100:.1f}%", flush=True)
    print("Optimal Parameters:", flush=True)
    print(f"  horizon: {best['horizon']}", flush=True)
    print(f"  max_waves: {best['max_waves']}", flush=True)
    print(f"  roi_threshold: {best['roi_threshold']:.4f}", flush=True)
    print(f"  min_ships: {best['min_ships']:.4f}", flush=True)
    print(f"  max_regroup_time: {best['max_regroup_time']:.4f}", flush=True)
    print(f"  regroup_pressure_delta: {best['regroup_pressure_delta']:.4f}", flush=True)
    
    # Clean up ref dir
    shutil.rmtree(v14_ref_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
