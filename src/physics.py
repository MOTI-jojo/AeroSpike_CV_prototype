import math
import random
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from .config import G, RHO, M_BALL, D_BALL, R_BALL, A_BALL, CL_DEFAULT, STROUHAL_NUMBER
from .input_handler import SimulationParams, ServeType

def solve_trajectory_3d(params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the 3D trajectory of the volleyball using scipy.integrate.solve_ivp.
    Returns (t, x, y, z, vx, vy, vz) arrays.
    """
    # Decompose initial velocity into 3 components:
    # alpha_deg = vertical angle (elevation)
    # azimuth_deg = horizontal aim angle (0 = straight, pos = right, neg = left)
    alpha_rad = math.radians(params.alpha_deg)
    azimuth_rad = math.radians(params.azimuth_deg)
    
    v_horizontal = params.v0 * math.cos(alpha_rad)
    v0_x = v_horizontal * math.cos(azimuth_rad)
    v0_y = params.v0 * math.sin(alpha_rad)
    v0_z = v_horizontal * math.sin(azimuth_rad)
    
    # State vector: [x, y, z, vx, vy, vz]
    initial_state = [0.0, params.y0, params.start_z, v0_x, v0_y, v0_z]
    
    # Angular velocity (for topspin)
    # Convert RPM to rad/s
    omega_rad = params.spin_rpm * 2 * math.pi / 60.0
    
    # Tilt of the spin axis (side-spin effect)
    # gamma = 0 means pure topspin (rotation around -Z axis)
    # gamma != 0 introduces rotation around Y axis, generating lateral Magnus force (Z axis)
    gamma = math.radians(params.spin_angle_deg)
    omega_vec = np.array([0.0, -omega_rad * math.sin(gamma), -omega_rad * math.cos(gamma)])
    
    # Random phase offsets for Karman vortex oscillations (float serve)
    phi_z = random.uniform(0, 2 * math.pi)
    phi_y = random.uniform(0, 2 * math.pi)
    
    def derivatives(t: float, state: np.ndarray) -> np.ndarray:
        x, y, z, vx, vy, vz = state
        v_vec = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v_vec)
        
        if v_mag == 0:
            return np.array([vx, vy, vz, 0.0, -G, 0.0])
        
        # Drag Force
        # F_drag = -0.5 * rho * Cd * A * v_mag * v_vec
        f_drag = -0.5 * RHO * params.cd * A_BALL * v_mag * v_vec
        
        # Lateral/Lift Force
        f_magnus = np.array([0.0, 0.0, 0.0])
        f_karman = np.array([0.0, 0.0, 0.0])
        
        # 1. Magnus Force (applies to ANY serve if there is spin)
        if omega_rad > 0:
            S_factor = (R_BALL * omega_rad) / v_mag if v_mag > 0 else 0
            cl_actual = CL_DEFAULT * S_factor # simplified relation
            omega_cross_v = np.cross(omega_vec, v_vec)
            norm_cross = np.linalg.norm(omega_cross_v)
            if norm_cross > 0:
                dir_magnus = omega_cross_v / norm_cross
                f_magnus = 0.5 * RHO * cl_actual * A_BALL * (v_mag**2) * dir_magnus
                
        # 2. Karman vortex street (applies only to FLOAT serve)
        if params.serve_type == ServeType.FLOAT:
            freq = (STROUHAL_NUMBER * v_mag / D_BALL) * 0.15 # ~2-3 Hz
            cl_z = 0.25 * math.sin(2 * math.pi * freq * t + phi_z)
            cl_y = 0.15 * math.sin(2 * math.pi * freq * t + phi_y)
            f_karman_z = 0.5 * RHO * cl_z * A_BALL * (v_mag**2)
            f_karman_y = 0.5 * RHO * cl_y * A_BALL * (v_mag**2)
            f_karman = np.array([0.0, f_karman_y, f_karman_z])
            
        f_lateral = f_magnus + f_karman
            
        # Total Force
        f_total = f_drag + f_lateral
        f_total[1] -= M_BALL * G # Gravity
        
        # Accelerations
        ax, ay, az = f_total / M_BALL
        
        return np.array([vx, vy, vz, ax, ay, az])
        
    # Event function to stop integration when ball hits the ground (y = 0)
    def hit_ground(t, state):
        return state[1] # y coordinate
    hit_ground.terminal = True
    hit_ground.direction = -1 # only trigger when going downwards
    
    # Time span (max 10 seconds flight)
    t_span = (0.0, 10.0)
    
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_state,
        method='RK45',
        events=hit_ground,
        dense_output=True,
        max_step=0.01 # ensure we don't miss oscillations in float serve
    )
    
    # Extract coordinates and velocities
    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    vx = sol.y[3]
    vy = sol.y[4]
    vz = sol.y[5]
    
    return t, x, y, z, vx, vy, vz

def calculate_impact_force(v0: float, m: float = M_BALL, dt: float = 0.01) -> float:
    """
    Simplified estimate of impact force (F * dt = m * dv)
    Assuming ball accelerates from 0 to v0 in dt seconds.
    """
    return (m * v0) / dt
