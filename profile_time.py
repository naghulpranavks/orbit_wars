import time
from kaggle_environments import make

def run():
    env = make("orbit_wars", configuration={"seed": 42})
    
    # We monkey-patch the agent to measure its turn time
    import buddy_model.main as main_agent
    
    class TimeProfilerAgent:
        def __init__(self):
            self.max_time = 0
            self.total_time = 0
            self.turns = 0
            self.agent = main_agent.agent
            
        def __call__(self, obs, config):
            t0 = time.time()
            res = self.agent(obs, config)
            t1 = time.time()
            dt = t1 - t0
            self.max_time = max(self.max_time, dt)
            self.total_time += dt
            self.turns += 1
            if dt > 0.5:
                print(f"Turn {self.turns} took {dt:.3f}s")
            return res
            
    profiler = TimeProfilerAgent()
    
    print("Starting simulation...")
    t0 = time.time()
    # run against itself or a simpler bot
    env.run([profiler, "submission_v14.tar.gz"])
    print(f"Total game time: {time.time()-t0:.2f}s")
    print(f"Max turn time: {profiler.max_time:.3f}s")
    print(f"Average turn time: {profiler.total_time/max(1, profiler.turns):.3f}s")

if __name__ == "__main__":
    run()
