import numpy as np
from scipy.optimize import minimize

def optimize_serve(base_params, target_x: float, target_z: float) -> dict:
    from aerospike_cv.core.physics import solve_trajectory_3d
    
    def objective(x_vars):
        v0, alpha, azimuth = x_vars
        p = base_params.copy()
        p.v0 = v0
        p.alpha_deg = alpha
        p.azimuth_deg = azimuth
        
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
        
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) == 0 or y[idx_net[0]] <= 2.43 + 0.105:
            dist += 50.0
            
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

def reverse_solve(base_params, target_x: float, target_z: float, target_speed_kmh: float = None) -> dict:
    from aerospike_cv.core.physics import solve_trajectory_3d
    
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
        
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) == 0 or y[idx_net[0]] <= 2.55:
            dist += 100.0
        
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
