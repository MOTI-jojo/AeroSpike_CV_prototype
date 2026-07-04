import os
import streamlit.components.v1 as components

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "volleyball_3d",
        url="http://localhost:5173",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(parent_dir, "..", "frontend", "dist"))
    _component_func = components.declare_component("volleyball_3d", path=build_dir)

def volleyball_3d(t, x, y, z, idx_max_v=0, key=None):
    """
    Shows a 3D volleyball court with the given trajectory.
    """
    component_value = _component_func(
        t=t.tolist() if hasattr(t, "tolist") else t,
        x=x.tolist() if hasattr(x, "tolist") else x,
        y=y.tolist() if hasattr(y, "tolist") else y,
        z=z.tolist() if hasattr(z, "tolist") else z,
        idx_max_v=idx_max_v,
        key=key,
        default=0
    )
    return component_value
