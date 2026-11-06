from kaggle_environments import make

env = make("orbit_wars")
trainer = env.train([None, "random"])
obs = trainer.reset()

print("Planet Array length:", len(obs["planets"][0]))
print("Planet Array sample:", obs["planets"][0])

if len(obs["fleets"]) > 0:
    print("Fleet Array length:", len(obs["fleets"][0]))
    print("Fleet Array sample:", obs["fleets"][0])
else:
    print("No fleets initially. Taking a step...")
    obs, _, _, _ = trainer.step([])
    if len(obs["fleets"]) > 0:
        print("Fleet Array sample:", obs["fleets"][0])
    else:
        # Step until a fleet appears
        for _ in range(50):
            action = trainer.act(obs) # Random action might launch
            obs, _, _, _ = trainer.step(action)
            if len(obs.get("fleets", [])) > 0:
                print("Fleet Array sample:", obs["fleets"][0])
                break
