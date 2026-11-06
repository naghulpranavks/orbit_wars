"""Download training output from a completed Kaggle kernel."""
import os
import subprocess
import shutil

KERNEL_SLUG = "naghulpranavks/notebook4617905c28"
OUTPUT_DIR = "kaggle_training_output"

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

if __name__ == "__main__":
    # 1. Check kernel status
    print("=" * 50)
    print("  Downloading Kaggle Training Output")
    print("=" * 50)
    
    run(f"kaggle kernels status {KERNEL_SLUG}")
    
    # 2. Download output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nDownloading output to {OUTPUT_DIR}/...")
    success = run(f"kaggle kernels output {KERNEL_SLUG} -p {OUTPUT_DIR}")
    
    if not success:
        print("\n[ERROR] Download failed. Is the kernel still running?")
        print("Output is only available after the kernel completes.")
        exit(1)
    
    # 3. Show what was downloaded
    print("\nDownloaded files:")
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
            rel = os.path.relpath(fpath, OUTPUT_DIR)
            print(f"  {rel} ({size / 1024:.1f} KB)")
    print(f"\nTotal: {total_size / (1024*1024):.1f} MB")
    
    # 4. Copy the final model to project root if it exists
    final_model = os.path.join(OUTPUT_DIR, "ppo_orbit_wars.zip")
    if os.path.exists(final_model):
        shutil.copy2(final_model, "ppo_orbit_wars.zip")
        print(f"\n[OK] Copied final model to ppo_orbit_wars.zip")
    
    # 5. Copy historical models if they exist
    hist_src = os.path.join(OUTPUT_DIR, "historical_models")
    if os.path.exists(hist_src):
        os.makedirs("historical_models", exist_ok=True)
        for f in os.listdir(hist_src):
            shutil.copy2(os.path.join(hist_src, f), os.path.join("historical_models", f))
        count = len(os.listdir(hist_src))
        print(f"[OK] Copied {count} historical models to historical_models/")
    
    print("\nDone! You can now use the trained model locally.")
