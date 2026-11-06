from stable_baselines3 import PPO
import numpy as np

print("Loading V9 AI Commander...")
model = PPO.load("buddy_model/v9_ai_commander.zip")
state = model.policy.state_dict()

print("Extracting Neural Network Weights...")
with open("buddy_model/v9_weights.py", "w") as f:
    f.write('"""\nPure Numpy Extraction of the V9 AI Commander.\nThis bypasses the Kaggle Tensorboard crash completely.\n"""\n')
    f.write("import numpy as np\n\n")
    
    # Layer 1 (64 neurons)
    f.write("W_IN = np.array(" + repr(state['mlp_extractor.policy_net.0.weight'].numpy().tolist()) + ")\n")
    f.write("b_IN = np.array(" + repr(state['mlp_extractor.policy_net.0.bias'].numpy().tolist()) + ")\n\n")
    
    # Layer 2 (64 neurons)
    f.write("W_H1 = np.array(" + repr(state['mlp_extractor.policy_net.2.weight'].numpy().tolist()) + ")\n")
    f.write("b_H1 = np.array(" + repr(state['mlp_extractor.policy_net.2.bias'].numpy().tolist()) + ")\n\n")
    
    # Output Layer (4 actions)
    f.write("W_OUT = np.array(" + repr(state['action_net.weight'].numpy().tolist()) + ")\n")
    f.write("b_OUT = np.array(" + repr(state['action_net.bias'].numpy().tolist()) + ")\n")

print("Extraction Complete! Weights saved to buddy_model/v9_weights.py")
