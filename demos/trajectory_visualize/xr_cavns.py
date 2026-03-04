import numpy as np
from simpub.core import XRCavns
import pyzlc

if __name__ == "__main__":
    cavns = XRCavns(ip_addr="192.168.0.117")

    num_points = 100
    x_values = np.linspace(0, 4 * np.pi, num_points)
    scale = 0.3

    def generate_waypoints(t: float):
        """Generate sin wave waypoints with gradient colors."""
        waypoints = []
        for i, x in enumerate(x_values):
            # Position: x along X-axis, sin(x - t) along Y-axis
            pos = [float(x * scale), float(np.sin(x - t) * scale), 0.0]
            # Color: gradient from red to blue based on position
            ratio = i / (num_points - 1)
            color = [1.0 - ratio, 0.0, ratio, 1.0]  # Red -> Blue
            waypoints.append({"pos": pos, "color": color})
        return waypoints

    # Initial trajectory with per-waypoint colors
    trajectory = cavns.create_trajectory(
        name="sin_wave",
        waypoints=generate_waypoints(0.0),
        width=0.02,
        resolution=10,
    )
    print("Created sin(x) trajectory with", num_points, "points")

    # Animate: traveling wave sin(x - t)
    t = 0.0
    dt = 0.1
    while True:
        t += dt
        trajectory.update(waypoints=generate_waypoints(t))
        pyzlc.sleep(0.05)  # 50ms between updates (~20 FPS)
