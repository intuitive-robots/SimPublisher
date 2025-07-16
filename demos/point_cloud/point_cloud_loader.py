from simpub.core.net_manager import init_node
from simpub.xr_device.xr_device import XRDevice
from simpub.xr_device.meta_quest3 import MetaQuest3
from simpub.core.net_manager import Publisher
import numpy as np


def generate_random_point_cloud(num_points=10000):
    xyz = np.random.uniform(-1, 1, (num_points, 3)).astype(np.float32)
    rgb = np.random.uniform(0, 1, (num_points, 3)).astype(np.float32)
    size = (np.ones((num_points, 1)) * 0.01).astype(np.float32)
    cloud = np.hstack((xyz, rgb, size))
    return cloud.astype(np.float32).tobytes()


if __name__ == "__main__":

    net_manager = init_node("192.168.178.60", "PointCloud") # 192.168.178.51
    net_manager.start_node_broadcast()
    unity_editor = MetaQuest3("UnityNode") # XRDevice("UnityNode")
    publisher = Publisher("PointCloud")
    try:
        while True:
            pointcloud_bytes = generate_random_point_cloud()
            publisher.publish_bytes(pointcloud_bytes)
    except KeyboardInterrupt:
        pass

    # data = simpublisher.generate_point_cloud_data(rgb_image, depth_image)
    # simpublisher.join()
