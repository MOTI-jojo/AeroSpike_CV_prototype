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
