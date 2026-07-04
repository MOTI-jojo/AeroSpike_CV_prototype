import numpy as np

def calculate_reception_zones(x: np.ndarray, y: np.ndarray, z: np.ndarray, time_arr: np.ndarray) -> dict:
    REACTION_TIME = 0.3  # секунды
    PLAYER_SPEED = 7.0   # м/с
    
    landing_idx_arr = np.where(y <= 0.15)[0]
    if len(landing_idx_arr) == 0:
        return {"error": "Ball never lands"}
    
    land_idx = landing_idx_arr[0]
    land_x, land_z, land_t = x[land_idx], z[land_idx], time_arr[land_idx]
    
    defenders = {
        "Pos 1 (Правый зад.)": (16.0, 3.0),
        "Pos 5 (Левый зад.)": (16.0, -3.0),
        "Pos 6 (Центр зад.)": (16.0, 0.0),
        "Либеро": (14.0, 0.0),
    }
    
    results = []
    for name, (dx, dz) in defenders.items():
        dist_to_ball = ((land_x - dx)**2 + (land_z - dz)**2)**0.5
        time_available = max(0, land_t - REACTION_TIME)
        reachable_dist = PLAYER_SPEED * time_available
        can_reach = reachable_dist >= dist_to_ball
        results.append({
            "name": name, "pos_x": dx, "pos_z": dz,
            "distance": dist_to_ball, "time_available": time_available,
            "can_reach": can_reach
        })
    
    return {"land_x": land_x, "land_z": land_z, "land_t": land_t, "defenders": results}
