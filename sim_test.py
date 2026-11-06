import torch
import time
from buddy_model.orbit_lite.movement import PlanetMovement
from buddy_model.orbit_lite.obs import parse_obs
import pickle

def test_simulation_speed():
    device = torch.device("cpu")
    # We need a dummy observation to parse. We can't easily mock this without the real environment.
    print("Test ready to be integrated into an actual environment loop to benchmark.")

if __name__ == "__main__":
    test_simulation_speed()
