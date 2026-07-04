from enum import Enum
from pydantic import BaseModel, Field

class ServeType(str, Enum):
    TOPSPIN = "topspin"
    FLOAT = "float"

class SimulationParams(BaseModel):
    """
    Validated input parameters for the simulation.
    """
    ball_type: str = Field(default="MIKASA_V200W", description="Type of the ball from config")
    mass: float = Field(default=0.27, gt=0, description="Mass of the ball in kg")
    
    # Initial conditions
    v0: float = Field(default=20.0, gt=0, description="Initial velocity module (m/s)")
    alpha_deg: float = Field(default=15.0, description="Launch angle in degrees (vertical)")
    azimuth_deg: float = Field(default=0.0, description="Horizontal aim angle (0=straight, neg=left, pos=right)")
    y0: float = Field(default=2.5, ge=0, description="Initial height (m)")
    start_z: float = Field(default=0.0, description="Lateral start position along baseline (m)")
    
    # Serve specifics
    serve_type: ServeType = Field(default=ServeType.TOPSPIN, description="Type of serve")
    spin_rpm: float = Field(default=800.0, ge=0, description="Spin rate in RPM (mostly for topspin)")
    spin_angle_deg: float = Field(default=0.0, description="Tilt of the spin axis in degrees (-45 to 45)")
    
    # Drag config (optional manual override)
    cd: float = Field(default=0.4, ge=0, description="Drag coefficient")
    
    # Wind
    wind_speed: float = Field(default=0.0, ge=0, description="Wind speed in m/s")
    wind_direction_deg: float = Field(default=0.0, description="Wind direction: 0=headwind(+x), 90=crosswind(+z), 180=tailwind(-x)")
