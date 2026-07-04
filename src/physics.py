import math
import random
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from .config import G, RHO, CL_DEFAULT, STROUHAL_NUMBER, BALL_MODELS, SPIN_DECAY_RATE, RESTITUTION_COEF_FLOOR, RESTITUTION_COEF_NET
from .input_handler import SimulationParams, ServeType

def solve_trajectory_3d(params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the 3D trajectory of the volleyball using scipy.integrate.solve_ivp.
    Returns (t, x, y, z, vx, vy, vz) arrays.
    Handles collisions (net, floor) by restarting integration.
    """
    ball = BALL_MODELS.get(params.ball_type, BALL_MODELS["CUSTOM"])
    m_ball = params.mass if params.ball_type == "CUSTOM" else ball.mass
    d_ball = ball.diameter
    r_ball = d_ball / 2
    a_ball = math.pi * (r_ball ** 2)
    cd_ball = params.cd if params.ball_type == "CUSTOM" else ball.cd

    alpha_rad = math.radians(params.alpha_deg)
    azimuth_rad = math.radians(params.azimuth_deg)
    
    v_horizontal = params.v0 * math.cos(alpha_rad)
    v0_x = v_horizontal * math.cos(azimuth_rad)
    v0_y = params.v0 * math.sin(alpha_rad)
    v0_z = v_horizontal * math.sin(azimuth_rad)
    
    current_state = [0.0, params.y0, params.start_z, v0_x, v0_y, v0_z]
    omega_rad = params.spin_rpm * 2 * math.pi / 60.0
    gamma = math.radians(params.spin_angle_deg)
    
    phi_z = random.uniform(0, 2 * math.pi)
    phi_y = random.uniform(0, 2 * math.pi)
    
    def derivatives(t: float, state: np.ndarray) -> np.ndarray:
        x, y, z, vx, vy, vz = state
        v_vec = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v_vec)
        
        if v_mag == 0:
            return np.array([vx, vy, vz, 0.0, -G, 0.0])
            
        f_drag = -0.5 * RHO * cd_ball * a_ball * v_mag * v_vec
        f_magnus = np.array([0.0, 0.0, 0.0])
        f_karman = np.array([0.0, 0.0, 0.0])
        
        # Spin decay over time
        curr_omega = omega_rad * math.exp(-SPIN_DECAY_RATE * t)
        omega_vec = np.array([0.0, -curr_omega * math.sin(gamma), -curr_omega * math.cos(gamma)])
        
        if curr_omega > 0:
            S_factor = (r_ball * curr_omega) / v_mag if v_mag > 0 else 0
            cl_actual = CL_DEFAULT * S_factor
            omega_cross_v = np.cross(omega_vec, v_vec)
            norm_cross = np.linalg.norm(omega_cross_v)
            if norm_cross > 0:
                dir_magnus = omega_cross_v / norm_cross
                f_magnus = 0.5 * RHO * cl_actual * a_ball * (v_mag**2) * dir_magnus
                
        if params.serve_type == ServeType.FLOAT:
            freq = (STROUHAL_NUMBER * v_mag / d_ball) * 0.15
            cl_z = 0.25 * math.sin(2 * math.pi * freq * t + phi_z)
            cl_y = 0.15 * math.sin(2 * math.pi * freq * t + phi_y)
            f_karman_z = 0.5 * RHO * cl_z * a_ball * (v_mag**2)
            f_karman_y = 0.5 * RHO * cl_y * a_ball * (v_mag**2)
            f_karman = np.array([0.0, f_karman_y, f_karman_z])
            
        f_lateral = f_magnus + f_karman
        f_total = f_drag + f_lateral
        f_total[1] -= m_ball * G
        
        ax, ay, az = f_total / m_ball
        return np.array([vx, vy, vz, ax, ay, az])

    def hit_ground(t, state):
        return state[1] - r_ball # y coordinate minus radius
    hit_ground.terminal = True
    hit_ground.direction = -1

    def hit_net(t, state):
        x, y = state[0], state[1]
        # Net is at x=9.0. If y < 2.43 (top of the net), it hits.
        if y < 2.43 + r_ball:
            return 9.0 - (x + r_ball) # Front of ball hits net
        return 1.0 # No collision
    hit_net.terminal = True
    hit_net.direction = -1
    
    t_start = 0.0
    t_end = 10.0
    
    # Store history
    history_t, history_x, history_y, history_z = [], [], [], []
    history_vx, history_vy, history_vz = [], [], []
    
    bounces = 0
    max_bounces = 3 # Stop after 3 bounces or impacts
    
    while t_start < t_end and bounces < max_bounces:
        sol = solve_ivp(
            fun=derivatives,
            t_span=(t_start, t_end),
            y0=current_state,
            method='RK45',
            events=[hit_ground, hit_net],
            dense_output=True,
            max_step=0.01
        )
        
        # Append to history
        history_t.append(sol.t)
        history_x.append(sol.y[0])
        history_y.append(sol.y[1])
        history_z.append(sol.y[2])
        history_vx.append(sol.y[3])
        history_vy.append(sol.y[4])
        history_vz.append(sol.y[5])
        
        if sol.status == 1: # A termination event occurred
            event_idx = -1
            for i, ev_t in enumerate(sol.t_events):
                if len(ev_t) > 0:
                    event_idx = i
                    break
            
            if event_idx == 0:
                # Hit ground
                current_state = sol.y[:, -1].copy()
                current_state[1] = r_ball # Correct position to exactly on ground
                # Reverse y velocity and apply restitution
                current_state[4] = -current_state[4] * RESTITUTION_COEF_FLOOR
                current_state[3] *= RESTITUTION_COEF_FLOOR # friction
                current_state[5] *= RESTITUTION_COEF_FLOOR
                
            elif event_idx == 1:
                # Hit net
                current_state = sol.y[:, -1].copy()
                current_state[0] = 9.0 - r_ball # correct position
                # Reverse x velocity (bounce off net back) and reduce greatly
                current_state[3] = -current_state[3] * RESTITUTION_COEF_NET
                current_state[4] *= 0.5 # loses vertical energy too
                current_state[5] *= RESTITUTION_COEF_NET
                
            t_start = sol.t[-1]
            bounces += 1
            
            # Stop if moving too slow after bounce
            if np.linalg.norm(current_state[3:6]) < 0.5:
                break
        else:
            break

    # Concatenate all phases
    t = np.concatenate(history_t)
    x = np.concatenate(history_x)
    y = np.concatenate(history_y)
    z = np.concatenate(history_z)
    vx = np.concatenate(history_vx)
    vy = np.concatenate(history_vy)
    vz = np.concatenate(history_vz)
    
    return t, x, y, z, vx, vy, vz

def calculate_impact_force(v0: float, m: float, dt: float = 0.01) -> float:
    """
    Simplified estimate of impact force (F * dt = m * dv)
    Assuming ball accelerates from 0 to v0 in dt seconds.
    """
    return (m * v0) / dt
