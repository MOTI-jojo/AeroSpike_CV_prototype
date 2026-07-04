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
    final_x = x[-1]
    final_z = z[-1]
    
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
        flight_time = time_arr[-1]
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
