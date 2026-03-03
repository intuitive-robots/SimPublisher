from simpub.core import XRCavns
import pyzlc

if __name__ == "__main__":
    cavns = XRCavns(ip_addr="192.168.0.117")
    trajectory = cavns.create_trajectory(
        name="test_trajectory",
        points=[[0, 0, 0], [1, 1, 0], [2, 0, 2]],
        color=[1, 0, 0, 1],
        width=0.05,
        resolution=10,
    )
    print("Created trajectory:", trajectory.name)
    print("Initial points:", trajectory.points)
    pyzlc.spin()