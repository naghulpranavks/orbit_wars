from kaggle_environments import make

# 1. Initialize the Orbit Wars environment
# We set debug=True to see any errors our agents might throw.
env = make("orbit_wars", configuration={"seed": 42}, debug=True)

# 2. Run a simulation between two agents.
# "random" is a built-in agent that just does random actions.
# Once we build our own agent in 'agent.py', we'll replace the first "random" with "agent.py".
print("Running simulation...")
env.run(["rl_agent.py", "agent.py"])

# 3. Render the output to an HTML file so we can watch the replay visually!
print("Simulation complete! Saving replay to replay.html...")
html = env.render(mode="html")
with open("replay.html", "w") as f:
    f.write(html)
print("Done! Open replay.html in your browser to watch the game.")
