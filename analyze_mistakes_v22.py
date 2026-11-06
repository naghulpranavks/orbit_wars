"""
Game Mistake Analyzer for V17.
Runs N games vs V15, saves per-turn state for losing games,
then identifies specific mistake patterns.
"""
import sys, os, shutil, tarfile, json, importlib, math
from collections import defaultdict

# ── extract reference bot ──────────────────────────────────────────────────
def extract(archive, d):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    with tarfile.open(archive) as t: t.extractall(d)

REF_DIR = os.path.abspath("v16_analysis_ref")
TEST_DIR = os.path.abspath("buddy_model")

V22_DIR = os.path.abspath("v22_bot")
V15_DIR = os.path.abspath("v15_bot")
V14_DIR = os.path.abspath("v14_bot")

extract("submission_v22.tar.gz", V22_DIR)
extract("submission_v15.tar.gz", V15_DIR)
extract("submission_v14.tar.gz", V14_DIR)

# ── load agents ────────────────────────────────────────────────────────────
def load_agent(d, label):
    sys.path.insert(0, d)
    for k in list(sys.modules):
        if "orbit_lite" in k or k == "main": sys.modules.pop(k, None)
    m = importlib.import_module("main")
    sys.path.remove(d)
    return m.agent, label

from kaggle_environments import make

# ── per-turn metrics collector ────────────────────────────────────────────
class TurnStats:
    def __init__(self, pid):
        self.pid = pid
        self.turns = []   # list of dicts

    def record(self, step, planets, fleets):
        pid = self.pid
        my_planets   = [p for p in planets if int(p[1]) == pid]
        opp_planets  = [p for p in planets if int(p[1]) not in (-1, pid) and int(p[1]) >= 0]
        neutral      = [p for p in planets if int(p[1]) == -1]
        my_ships     = sum(float(p[5]) for p in my_planets)
        opp_ships    = sum(float(p[5]) for p in opp_planets)
        my_prod      = sum(float(p[7]) for p in my_planets) if my_planets and len(my_planets[0]) > 7 else 0
        opp_prod     = sum(float(p[7]) for p in opp_planets) if opp_planets and len(opp_planets[0]) > 7 else 0
        # in-flight ships belonging to us vs opponent
        my_fleet_ships  = sum(float(f[3]) for f in fleets if int(f[1]) == pid) if fleets else 0
        opp_fleet_ships = sum(float(f[3]) for f in fleets if int(f[1]) not in (-1, pid) and int(f[1]) >= 0) if fleets else 0
        self.turns.append({
            "step": step,
            "n_mine": len(my_planets),
            "n_opp": len(opp_planets),
            "n_neutral": len(neutral),
            "my_ships": my_ships,
            "opp_ships": opp_ships,
            "my_fleet": my_fleet_ships,
            "opp_fleet": opp_fleet_ships,
            "my_total": my_ships + my_fleet_ships,
            "opp_total": opp_ships + opp_fleet_ships,
            "my_prod": my_prod,
            "opp_prod": opp_prod,
        })

def run_game_and_capture(seed, test_agent, agent_v22, agent_v15, agent_v14, test_pid=0):
    """Run a game and capture per-turn snapshots."""
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    snapshots = []   # (step, planets, fleets)

    # Monkey-patch step-level observation capture
    agents = [test_agent, agent_v22, agent_v15, agent_v14]
    result = env.run(agents)

    for step_idx, step in enumerate(result):
        obs = step[0].get("observation", {})
        planets = obs.get("planets", [])
        fleets  = obs.get("fleets", [])
        snapshots.append((step_idx, planets, fleets))

    final = result[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    if r0 is None or r1 is None:
        return "DRAW", snapshots
    if r0 > r1:
        return ("WIN" if test_pid == 0 else "LOSS"), snapshots
    elif r0 < r1:
        return ("LOSS" if test_pid == 0 else "WIN"), snapshots
    return "DRAW", snapshots

# ── mistake detectors ─────────────────────────────────────────────────────
def find_planet_losses(snapshots, pid):
    """Find turns where we lost a planet (owned → not owned)."""
    losses = []
    prev_mine = set()
    for step_idx, planets, fleets in snapshots:
        mine = set(int(p[0]) for p in planets if int(p[1]) == pid)
        lost = prev_mine - mine
        for pid_lost in lost:
            losses.append({"step": step_idx, "planet_id": pid_lost})
        prev_mine = mine
    return losses

def find_idle_ship_hoarding(snapshots, pid, threshold=200):
    """Turns where we have a single planet with >threshold ships and aren't attacking."""
    hoard_turns = []
    for step_idx, planets, fleets in snapshots:
        my_planets = [p for p in planets if int(p[1]) == pid]
        for p in my_planets:
            if float(p[5]) > threshold:
                my_fleets_this_turn = [f for f in fleets if int(f[1]) == pid]
                if len(my_fleets_this_turn) == 0:
                    hoard_turns.append({"step": step_idx, "planet_id": int(p[0]), "ships": float(p[5])})
    return hoard_turns

def find_ship_advantage_loss(snapshots, pid):
    """Turns where our ship advantage REVERSED (we went from leading to trailing)."""
    reversals = []
    was_ahead = None
    for step_idx, planets, fleets in snapshots:
        my_ships  = sum(float(p[5]) for p in planets if int(p[1]) == pid)
        opp_ships = sum(float(p[5]) for p in planets if int(p[1]) not in (-1, pid) and int(p[1]) >= 0)
        my_fleets  = sum(float(f[3]) for f in fleets if int(f[1]) == pid)
        opp_fleets = sum(float(f[3]) for f in fleets if int(f[1]) not in (-1, pid) and int(f[1]) >= 0)
        total_mine = my_ships + my_fleets
        total_opp  = opp_ships + opp_fleets
        now_ahead = total_mine > total_opp
        if was_ahead is True and not now_ahead:
            reversals.append({"step": step_idx, "my_total": total_mine, "opp_total": total_opp})
        was_ahead = now_ahead
    return reversals

def find_slow_expansion(snapshots, pid):
    """Check if we captured fewer than 2 neutrals in first 80 turns."""
    initial_mine = 0
    captured = 0
    prev_mine = set()
    for step_idx, planets, fleets in snapshots:
        if step_idx > 80: break
        mine = set(int(p[0]) for p in planets if int(p[1]) == pid)
        if step_idx == 0:
            initial_mine = len(mine)
        gained = mine - prev_mine
        prev_mine = mine
        captured += len(gained)
    return max(0, captured - initial_mine)  # neutrals captured (minus initial own)

def find_wasted_attacks(snapshots, pid):
    """Planets we captured but lost within 15 turns → wasted attack."""
    capture_turn = {}   # planet_id → turn captured
    wasted = []
    prev_mine = set()
    for step_idx, planets, fleets in snapshots:
        mine = set(int(p[0]) for p in planets if int(p[1]) == pid)
        gained = mine - prev_mine
        lost   = prev_mine - mine
        for pg in gained:
            capture_turn[pg] = step_idx
        for pl in lost:
            if pl in capture_turn and (step_idx - capture_turn[pl]) <= 15:
                wasted.append({"planet_id": pl, "captured_at": capture_turn[pl], "lost_at": step_idx, "held_for": step_idx - capture_turn[pl]})
        prev_mine = mine
    return wasted

def find_late_game_collapse(snapshots, pid):
    """Check if we had a big ship lead after turn 200 but still lost."""
    had_lead = False
    for step_idx, planets, fleets in snapshots:
        if step_idx < 200: continue
        my    = sum(float(p[5]) for p in planets if int(p[1]) == pid)
        opp   = sum(float(p[5]) for p in planets if int(p[1]) not in (-1, pid) and int(p[1]) >= 0)
        if my > opp * 1.3:
            had_lead = True
            break
    return had_lead

# ── main loop ────────────────────────────────────────────────────────────
N_GAMES = 1
SEEDS = list(range(3000, 3000 + N_GAMES))

print(f"Running {N_GAMES} games (V23 vs V22 vs V15 vs V14) and capturing per-turn data...")
test_agent, _ = load_agent(TEST_DIR, "V23")
agent_v22, _ = load_agent(V22_DIR, "V22")
agent_v15, _ = load_agent(V15_DIR, "V15")
agent_v14, _ = load_agent(V14_DIR, "V14")

mistake_counts = defaultdict(int)
mistake_examples = defaultdict(list)
results = []

for i, seed in enumerate(SEEDS):
    outcome, snaps = run_game_and_capture(seed, test_agent, agent_v22, agent_v15, agent_v14, test_pid=0)
    results.append(outcome)
    print(f"  Game {i+1}/{N_GAMES} seed={seed}: {outcome}", flush=True)

    if outcome == "LOSS":
        pid = 0
        # Run all detectors
        losses = find_planet_losses(snaps, pid)
        hoards = find_idle_ship_hoarding(snaps, pid)
        reversals = find_ship_advantage_loss(snaps, pid)
        neutrals_captured = find_slow_expansion(snaps, pid)
        wasted = find_wasted_attacks(snaps, pid)
        late_collapse = find_late_game_collapse(snaps, pid)

        if len(losses) > 5:
            mistake_counts["many_planet_losses"] += 1
            mistake_examples["many_planet_losses"].append({"seed": seed, "losses": len(losses)})

        if hoards:
            mistake_counts["ship_hoarding"] += 1
            mistake_examples["ship_hoarding"].append({"seed": seed, "hoard_turns": len(hoards), "example": hoards[0]})

        if reversals:
            mistake_counts["ship_advantage_squandered"] += 1
            mistake_examples["ship_advantage_squandered"].append({"seed": seed, "first_reversal_turn": reversals[0]["step"]})

        if neutrals_captured < 2:
            mistake_counts["slow_expansion"] += 1
            mistake_examples["slow_expansion"].append({"seed": seed, "neutrals_in_80_turns": neutrals_captured})

        if wasted:
            mistake_counts["wasted_attacks"] += 1
            mistake_examples["wasted_attacks"].append({"seed": seed, "count": len(wasted), "example": wasted[0]})

        if late_collapse:
            mistake_counts["late_game_collapse"] += 1
            mistake_examples["late_game_collapse"].append({"seed": seed})

wins   = results.count("WIN")
losses = results.count("LOSS")
draws  = results.count("DRAW")
print(f"\n{'='*55}")
print(f"RESULTS: {wins}W / {losses}L / {draws}D  ({wins/N_GAMES*100:.0f}%)")
print(f"{'='*55}")
print(f"\nMISTAKE ANALYSIS (from {losses} losing games):")
print(f"{'='*55}")
MISTAKE_LABELS = {
    "many_planet_losses":         "Planet chain-losses (lost 5+ planets in one game)",
    "ship_hoarding":              "Ship hoarding (200+ idle ships, no fleets launched)",
    "ship_advantage_squandered":  "Ship advantage squandered (led then fell behind)",
    "slow_expansion":             "Slow early expansion (<2 neutrals in first 80 turns)",
    "wasted_attacks":             "Wasted attacks (captured planet, lost it within 15 turns)",
    "late_game_collapse":         "Late-game collapse (led after turn 200, still lost)",
}
for key, label in MISTAKE_LABELS.items():
    count = mistake_counts[key]
    pct = count / max(losses, 1) * 100
    bar = "#" * count + "-" * max(0, losses - count)
    print(f"\n  {label}")
    print(f"  [{bar}]  {count}/{losses} losing games ({pct:.0f}%)")
    for ex in mistake_examples[key][:2]:
        print(f"    -> {ex}")

print(f"\n{'='*55}")
print("DIAGNOSIS SUMMARY")
print(f"{'='*55}")
top = sorted(mistake_counts.items(), key=lambda x: -x[1])
for k, v in top:
    print(f"  #{top.index((k,v))+1}  {MISTAKE_LABELS[k]}  ({v} games)")

shutil.rmtree(REF_DIR, ignore_errors=True)
