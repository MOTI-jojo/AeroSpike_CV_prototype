import numpy as np
from typing import List, Dict
from .input_handler import ServeType

def evaluate_serve(x: np.ndarray, y: np.ndarray, z: np.ndarray, time_arr: np.ndarray, speed_arr: np.ndarray, serve_type: ServeType, t: dict) -> List[Dict[str, str]]:
    """
    Evaluates the serve trajectory against volleyball rules and physical effectiveness.
    Returns a list of dictionaries with keys: 'status', 'param', 'comment'.
    'status' can be 'success', 'error', 'warning'.
    """
    results = []
    
    # 1. Net Clearance
    net_x = 9.0
    net_y = 2.43
    
    crossed_net = False
    hit_net = False
    
    # Find index where ball reaches the net
    idx_net = np.where(x >= net_x)[0]
    if len(idx_net) > 0:
        first_idx = idx_net[0]
        if y[first_idx] > net_y:
            crossed_net = True
            results.append({"status": "success", "param": t["an_net_param"], "comment": t["an_net_ok"]})
        else:
            hit_net = True
            results.append({"status": "error", "param": t["an_net_param"], "comment": t["an_net_fail"]})
    else:
        # Never reached 9 meters
        results.append({"status": "error", "param": t["an_net_param"], "comment": t["an_net_short"]})
        
    # 2. Out of Bounds (Landing zone)
    landing_idx = np.where(y <= 0.15)[0]
    if len(landing_idx) > 0:
        first_land = landing_idx[0]
        final_x = x[first_land]
        final_z = z[first_land]
        flight_time = time_arr[first_land]
    else:
        final_x = x[-1]
        final_z = z[-1]
        flight_time = time_arr[-1]
    
    if crossed_net:
        if final_x > 18.0:
            results.append({"status": "error", "param": t["an_out_param"], "comment": t["an_out_long"]})
        elif abs(final_z) > 4.5:
            results.append({"status": "error", "param": t["an_out_param"], "comment": t["an_out_wide"]})
        elif 17.0 <= final_x <= 18.0:
            results.append({"status": "success", "param": t["an_out_param"], "comment": t["an_out_ace"]})
        else:
            results.append({"status": "success", "param": t["an_out_param"], "comment": t["an_out_in"]})
            
    # 3. Flight Time / Reaction Speed
    if crossed_net and not hit_net and final_x <= 18.0 and abs(final_z) <= 4.5:
        if flight_time < 1.2:
            results.append({"status": "success", "param": t["an_time_param"], "comment": t["an_time_fast"]})
        elif flight_time > 1.5:
            results.append({"status": "warning", "param": t["an_time_param"], "comment": t["an_time_slow"]})
        else:
            results.append({"status": "success", "param": t["an_time_param"], "comment": t["an_time_normal"]})
            
        # 4. Special behaviors
        if serve_type == ServeType.FLOAT:
            max_drift = np.max(np.abs(z))
            if max_drift > 0.3:
                results.append({"status": "success", "param": t["an_drift_param"], "comment": t["an_drift_good"]})
        
        if serve_type == ServeType.TOPSPIN:
            # Evaluate Magnus effectiveness by looking at the final drop angle (steepness of the dive)
            # A strong topspin will cause the ball to dive sharply at the end.
            if len(time_arr) >= 2:
                v_y_final = (y[-1] - y[-2]) / (time_arr[-1] - time_arr[-2])
                v_x_final = (x[-1] - x[-2]) / (time_arr[-1] - time_arr[-2])
                drop_ratio = abs(v_y_final / v_x_final) if v_x_final != 0 else 1.0
                
                # A drop_ratio > 0.6 corresponds to a drop angle of > ~31 degrees, which is steep for a spike serve
                if drop_ratio > 0.6:
                    results.append({"status": "success", "param": t["an_magnus_param"], "comment": t["an_magnus_strong"]})
                else:
                    results.append({"status": "warning", "param": t["an_magnus_param"], "comment": t["an_magnus_weak"]})
                
    return results

import copy
import random
from typing import Tuple

def run_monte_carlo(params, n=30) -> Tuple[float, List[Tuple[float, float, bool]]]:
    """
    Runs N simulations with Gaussian noise on input parameters.
    Returns the success percentage and a list of landing points (x, z, is_success).
    """
    from .physics import solve_trajectory_3d
    points = []
    successes = 0
    
    for _ in range(n):
        p = params.copy()
        p.v0 = params.v0 * random.gauss(1.0, 0.05) # 5% noise on speed
        p.alpha_deg = params.alpha_deg + random.gauss(0, 2.0) # 2 degrees noise on vertical angle
        p.azimuth_deg = params.azimuth_deg + random.gauss(0, 1.5) # 1.5 degrees noise on horizontal angle
        
        t, x, y, z, vx, vy, vz = solve_trajectory_3d(p)
        
        crossed_net = False
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) > 0 and y[idx_net[0]] > 2.43:
            crossed_net = True
            
        landing_idx = np.where(y <= 0.15)[0]
        if len(landing_idx) > 0:
            first_land = landing_idx[0]
            final_x = x[first_land]
            final_z = z[first_land]
        else:
            final_x = x[-1]
            final_z = z[-1]
        
        is_success = crossed_net and (9.0 < final_x <= 18.0) and (abs(final_z) <= 4.5)
        
        if is_success:
            successes += 1
            
        points.append((final_x, final_z, is_success))
        
    return (successes / n) * 100.0, points

def optimize_serve(base_params, target_x: float, target_z: float) -> dict:
    """
    Finds optimal v0, alpha, azimuth to hit (target_x, target_z).
    Returns a dict with the optimal parameters.
    """
    from .physics import solve_trajectory_3d
    from scipy.optimize import minimize
    
    def objective(x_vars):
        v0, alpha, azimuth = x_vars
        p = base_params.copy()
        p.v0 = v0
        p.alpha_deg = alpha
        p.azimuth_deg = azimuth
        
        # Penalize unphysical params
        if v0 < 10 or v0 > 40: return 1000.0
        if alpha < -5 or alpha > 45: return 1000.0
        if azimuth < -30 or azimuth > 30: return 1000.0
        
        t, x, y, z, vx, vy, vz = solve_trajectory_3d(p)
        landing_idx = np.where(y <= 0.15)[0]
        if len(landing_idx) > 0:
            first_land = landing_idx[0]
            final_x = x[first_land]
            final_z = z[first_land]
        else:
            final_x = x[-1]
            final_z = z[-1]
        
        dist = ((final_x - target_x)**2 + (final_z - target_z)**2)**0.5
        
        # Check net
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) == 0 or y[idx_net[0]] <= 2.43 + 0.105: # net height + ball radius approx
            dist += 50.0 # Penalty for hitting net
            
        return dist
        
    initial_guess = [base_params.v0, base_params.alpha_deg, base_params.azimuth_deg]
    res = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 40})
    
    return {
        "v0": res.x[0],
        "alpha_deg": res.x[1],
        "azimuth_deg": res.x[2],
        "success": res.fun < 2.0,
        "dist": res.fun
    }

def calculate_reception_zones(x: np.ndarray, y: np.ndarray, z: np.ndarray, time_arr: np.ndarray) -> dict:
    """
    Рассчитывает, могут ли защитники на стандартных позициях добежать до точки падения мяча.
    """
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


def reverse_solve(base_params, target_x: float, target_z: float, target_speed_kmh: float = None) -> dict:
    """
    Обратная задача: по заданной точке приземления (и опционально скорости прилёта)
    подбирает параметры подачи (v0, alpha, azimuth).
    """
    from .physics import solve_trajectory_3d
    from scipy.optimize import minimize
    
    def objective(x_vars):
        v0, alpha, azimuth = x_vars
        p = base_params.copy()
        p.v0 = v0
        p.alpha_deg = alpha
        p.azimuth_deg = azimuth
        
        if v0 < 5 or v0 > 45: return 1e6
        if alpha < -10 or alpha > 50: return 1e6
        if azimuth < -35 or azimuth > 35: return 1e6
        
        try:
            t, x, y, z, vx, vy, vz = solve_trajectory_3d(p)
        except:
            return 1e6
            
        landing_idx_arr = np.where(y <= 0.15)[0]
        if len(landing_idx_arr) == 0: return 1e6
        li = landing_idx_arr[0]
        fx, fz = x[li], z[li]
        
        dist = ((fx - target_x)**2 + (fz - target_z)**2)**0.5
        
        # Штраф за попадание в сетку
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) == 0 or y[idx_net[0]] <= 2.55:
            dist += 100.0
        
        # Штраф за несоответствие скорости прилёта
        if target_speed_kmh is not None:
            speed_at_land = ((vx[li]**2 + vy[li]**2 + vz[li]**2)**0.5) * 3.6
            dist += abs(speed_at_land - target_speed_kmh) * 0.1
        
        return dist
    
    initial_guess = [base_params.v0, base_params.alpha_deg, base_params.azimuth_deg]
    res = minimize(objective, initial_guess, method='Nelder-Mead', 
                   options={'maxiter': 80, 'xatol': 0.01, 'fatol': 0.01})
    
    return {
        "v0": res.x[0], "alpha_deg": res.x[1], "azimuth_deg": res.x[2],
        "v0_kmh": res.x[0] * 3.6, "success": res.fun < 2.0, "dist_error": res.fun
    }
