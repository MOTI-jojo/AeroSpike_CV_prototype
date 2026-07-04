from enum import Enum
from pydantic import BaseModel, Field

class ServeType(str, Enum):
    TOPSPIN = "topspin"
    FLOAT = "float"

class SimulationParams(BaseModel):
    """
    Validated input parameters for the simulation.
    """
    mass: float = Field(default=0.27, gt=0, description="Mass of the ball in kg")
    diameter: float = Field(default=0.21, gt=0, description="Diameter of the ball in m")
    
    # Initial conditions
    v0: float = Field(default=20.0, gt=0, description="Initial velocity module (m/s)")
    alpha_deg: float = Field(default=15.0, description="Launch angle in degrees (vertical)")
    azimuth_deg: float = Field(default=0.0, description="Horizontal aim angle (0=straight, neg=left, pos=right)")
    y0: float = Field(default=2.5, ge=0, description="Initial height (m)")
    start_z: float = Field(default=0.0, description="Lateral start position along baseline (m)")
    
    # Serve specifics
    serve_type: ServeType = Field(default=ServeType.TOPSPIN, description="Type of serve")
    spin_rpm: float = Field(default=600.0, ge=0, description="Spin rate in RPM (mostly for topspin)")
    
    # Drag config (optional manual override)
    cd: float = Field(default=0.4, ge=0, description="Drag coefficient")
