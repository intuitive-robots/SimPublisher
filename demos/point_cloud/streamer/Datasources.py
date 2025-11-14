from dataclasses import dataclass

@dataclass
class CameraConfig:
    # Default from a realsense camera with resolution 640x480
    fx: float = 591.4252
    fy: float = 591.4252
    cx: float = 320.1326
    cy: float = 239.1477

