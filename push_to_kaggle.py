import os
import json
import subprocess

DATASET_SLUG = "orbit-wars-self-play-code"
# Replace with your actual Kaggle username!
# The Kaggle API requires the username in the dataset identifier.
USERNAME = "naghulpranavks"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0

def push_to_kaggle():
    metadata_path = "dataset-metadata.json"
    
    # Check if metadata exists
    if not os.path.exists(metadata_path):
        print("Initializing Kaggle dataset metadata...")
        # We create it manually to ensure correct formatting
        metadata = {
          "title": "Orbit Wars Self Play Code",
          "id": f"{USERNAME}/{DATASET_SLUG}",
          "licenses": [
            {
              "name": "CC0-1.0"
            }
          ]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print("Creating new Kaggle dataset...")
        run_cmd("kaggle datasets create -p .")
    else:
        print("Pushing new version to Kaggle dataset...")
        run_cmd('kaggle datasets version -p . -m "Automated code update"')

if __name__ == "__main__":
    print("=== Kaggle Automated Deployment ===")
    print("Make sure you have your kaggle.json authenticated!")
    push_to_kaggle()
    print("Done! Your code is now available on Kaggle.")
