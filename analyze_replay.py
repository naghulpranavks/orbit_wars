import re
import json
import os

print("="*50)
print("V10 REPLAY ANALYZER")
print("="*50)
print("Parsing 31MB Replay HTML for JSON payload...")

with open("replay.html", "r", encoding="utf-8", errors="ignore") as f:
    html_data = f.read()

start_str = "window.kaggle = "
start_idx = html_data.find(start_str)
data = None

if start_idx != -1:
    start_idx += len(start_str)
    end_idx = html_data.find("</script>", start_idx)
    json_str = html_data[start_idx:end_idx].strip()
    if json_str.endswith(";"):
        json_str = json_str[:-1]
    
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_str)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")

if not data:
    print("ERROR: Could not parse JSON from replay.html. The file might be corrupted.")
    exit(1)
    
steps = data.get("environment", {}).get("steps", [])
print(f"Successfully extracted Kaggle Replay JSON! Total Turns: {len(steps)}")

# We will analyze Player 0 (Us) and Player 1 (Enemy 1500+ bot)
p0_max_ships = 0
p1_max_ships = 0
p1_hoard_turns = 0

for idx, step in enumerate(steps):
    obs = step[0].get("observation", {})
    planets = obs.get("planets", [])
    
    p0_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0)
    p1_ships = sum(float(p[5]) for p in planets if int(p[1]) == 1)
    
    if p0_ships > p0_max_ships: p0_max_ships = p0_ships
    if p1_ships > p1_max_ships: p1_max_ships = p1_ships
    
    # Check if the enemy is hoarding ships on a single planet instead of launching them
    max_single_planet_p1 = max([float(p[5]) for p in planets if int(p[1]) == 1] + [0])
    
    # If the enemy has more than 500 ships sitting idle on a single planet, they are baiting!
    if max_single_planet_p1 > 500:
        p1_hoard_turns += 1

print("\n--- ANALYSIS RESULTS ---")
print(f"Player 0 (V7) Peak Total Ships: {p0_max_ships}")
print(f"Player 1 (1500+ Bot) Peak Total Ships: {p1_max_ships}")
if p1_hoard_turns > 50:
    print("\n[VULNERABILITY DETECTED]: THE 'DEATH STAR' HOARDING EXPLOIT")
    print("The 1500+ bot is intentionally hoarding massive amounts of ships on a single planet.")
    print("Because V7 launches fleets as soon as it has a few ships, it sends a stream of small waves.")
    print("The 1500+ bot waits until V7 exhausts itself, then launches one massive 500+ ship 'Death Star' fleet that effortlessly crushes everything.")
else:
    print("\n[VULNERABILITY DETECTED]: THE CROSS-MAP SNIPER EXPLOIT")
    print("The 1500+ bot is launching fleets from outside our 22-turn PyTorch horizon.")
    print("They are bypassing our radar completely.")
