import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

def agent(obs):
    """
    A simple greedy agent for Orbit Wars.
    It looks at all planets it owns. If a planet has more than 20 ships, 
    it finds the weakest unowned planet and launches half of its ships towards it.
    """
    # Parse the observation
    planets = [Planet(*p) for p in obs.get("planets", [])]
    player_id = obs.get("player", 0)
    
    # Separate planets by ownership
    my_planets = [p for p in planets if p.owner == player_id]
    other_planets = [p for p in planets if p.owner != player_id]
    
    actions = []
    
    # If there are no targets left, do nothing
    if not other_planets:
        return actions
        
    for p in my_planets:
        # Only launch if we have a decent garrison to defend our own planet
        if p.ships > 20:
            # Find the weakest target planet (least ships)
            weakest_target = min(other_planets, key=lambda target: target.ships)
            
            # Calculate the angle from our planet to the target
            # Note: Math.atan2 takes (y, x), but usually we use differences
            dy = weakest_target.y - p.y
            dx = weakest_target.x - p.x
            angle = math.atan2(dy, dx)
            
            # Send half of our available ships
            num_ships_to_send = p.ships // 2
            
            # Append action: [from_planet_id, direction_angle, num_ships]
            actions.append([p.id, angle, num_ships_to_send])
            
    return actions
