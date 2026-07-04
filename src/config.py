"""
Physical constants and default configurations for AeroSpike_CV_prototype.
"""
import math
from dataclasses import dataclass

# Environmental constants
G = 9.81  # Acceleration due to gravity (m/s^2)
RHO = 1.225  # Air density at sea level and 15°C (kg/m^3)

@dataclass
class BallModel:
    name: str
    mass: float
    diameter: float
    cd: float

BALL_MODELS = {
    "MIKASA_V200W": BallModel("Mikasa V200W", 0.27, 0.21, 0.40),
    "MIKASA_BEACH_PRO": BallModel("Mikasa Beach Pro", 0.27, 0.215, 0.42),
    "CUSTOM": BallModel("Custom", 0.27, 0.21, 0.40)
}

# Ball constants (Default fallback)
M_BALL = BALL_MODELS["MIKASA_V200W"].mass
D_BALL = BALL_MODELS["MIKASA_V200W"].diameter
R_BALL = D_BALL / 2
A_BALL = math.pi * (R_BALL ** 2)

# Aerodynamic constants
CD_DEFAULT = BALL_MODELS["MIKASA_V200W"].cd
CL_DEFAULT = 0.3 # Varies with spin parameter S = R*omega / v
STROUHAL_NUMBER = 0.2

# Physics extensions (Stage 1)
SPIN_DECAY_RATE = 0.5  # 1/s, exponential decay of spin
RESTITUTION_COEF_FLOOR = 0.65  # Velocity preserved after bounce
RESTITUTION_COEF_NET = 0.1  # Velocity preserved after hitting net (net absorbs most energy)

@dataclass
class ProPreset:
    name: str
    description: str
    v0_kmh: float
    alpha_deg: float
    azimuth_deg: float
    y0: float
    spin_rpm: float
    spin_angle_deg: float
    serve_type: str  # 'topspin' or 'float'

PRO_PRESETS = {
    "leon": ProPreset("Вильфредо Леон", "Мощный силовой топспин — рекорд 135 км/ч", 130.0, 8.0, 5.0, 3.3, 1200.0, 0.0, "topspin"),
    "ngapeth": ProPreset("Эрвин Нганпет", "Планирующая подача с обманным вращением", 85.0, 12.0, -3.0, 2.8, 50.0, 10.0, "float"),
    "anderson": ProPreset("Мэтт Андерсон", "Стабильная силовая подача со средним вращением", 110.0, 10.0, 0.0, 3.1, 900.0, -5.0, "topspin"),
    "lucarelli": ProPreset("Рикардо Лукарелли", "Быстрая атакующая подача в зону 1", 120.0, 7.0, 12.0, 3.2, 1100.0, 5.0, "topspin"),
    "earvin": ProPreset("Нганпет (Флот)", "Классическая планирующая - минимум вращения", 75.0, 14.0, 0.0, 2.7, 20.0, 0.0, "float"),
}
