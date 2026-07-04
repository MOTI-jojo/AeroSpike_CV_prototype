import numpy as np
import random
from typing import List, Dict, Tuple
from aerospike_cv.core.models import ServeType

def evaluate_serve(x: np.ndarray, y: np.ndarray, z: np.ndarray, time_arr: np.ndarray, speed_arr: np.ndarray, serve_type: ServeType, t: dict) -> List[Dict[str, str]]:
    results = []
    
    net_x = 9.0
    net_y = 2.43
    
    crossed_net = False
    hit_net = False
    
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
        results.append({"status": "error", "param": t["an_net_param"], "comment": t["an_net_short"]})
        
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
            
    if crossed_net and not hit_net and final_x <= 18.0 and abs(final_z) <= 4.5:
        if flight_time < 1.2:
            results.append({"status": "success", "param": t["an_time_param"], "comment": t["an_time_fast"]})
        elif flight_time > 1.5:
            results.append({"status": "warning", "param": t["an_time_param"], "comment": t["an_time_slow"]})
        else:
            results.append({"status": "success", "param": t["an_time_param"], "comment": t["an_time_normal"]})
            
        if serve_type == ServeType.FLOAT:
            max_drift = np.max(np.abs(z))
            if max_drift > 0.3:
                results.append({"status": "success", "param": t["an_drift_param"], "comment": t["an_drift_good"]})
        
        if serve_type == ServeType.TOPSPIN:
            if len(time_arr) >= 2:
                v_y_final = (y[-1] - y[-2]) / (time_arr[-1] - time_arr[-2])
                v_x_final = (x[-1] - x[-2]) / (time_arr[-1] - time_arr[-2])
                drop_ratio = abs(v_y_final / v_x_final) if v_x_final != 0 else 1.0
                
                if drop_ratio > 0.6:
                    results.append({"status": "success", "param": t["an_magnus_param"], "comment": t["an_magnus_strong"]})
                else:
                    results.append({"status": "warning", "param": t["an_magnus_param"], "comment": t["an_magnus_weak"]})
                
    return results

def run_monte_carlo(params, n=30) -> Tuple[float, List[Tuple[float, float, bool]]]:
    from aerospike_cv.core.physics import solve_trajectory_3d
    points = []
    successes = 0
    
    for _ in range(n):
        p = params.copy()
        p.v0 = params.v0 * random.gauss(1.0, 0.05)
        p.alpha_deg = params.alpha_deg + random.gauss(0, 2.0)
        p.azimuth_deg = params.azimuth_deg + random.gauss(0, 1.5)
        
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
