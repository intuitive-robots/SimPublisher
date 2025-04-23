import random
import cv2

import json
from simpub.core.net_manager import init_node
from simpub.xr_device.xr_device import XRDevice


def generate_point_cloud_data(rgb_image, depth_image):
    positions = []
    colors = []
    height, width, _ = rgb_image.shape
    for y in range(height):
        for x in range(width):
            rgb_pixel = rgb_image[y, x]
            depth_pixel = depth_image[y, x]
            if random.random() > 0.1:
                continue
            positions.extend([x / width, y / height, depth_pixel[0] / 10])
            colors.extend([int(rgb_pixel[0]) / 255, int(rgb_pixel[1]) / 255, int(rgb_pixel[2]) / 255, 1])
    return json.dumps({"positions": positions, "colors": colors})

rgb_path = '/home/xinkai/project/point-cloud-labeling/rgb.png'
depth_path = '/home/xinkai/project/point-cloud-labeling/depth.png'

rgb_image = cv2.imread(rgb_path)
depth_image = cv2.imread(depth_path)

# simpublisher = RGBDPublisher(rgb_image, depth_image)
net_manager = init_node("127.0.0.1", "PointCloudPublisher")
unity_editor = XRDevice()
# print(generate_point_cloud_data(rgb_image, depth_image))
while unity_editor.connected is False:
    pass
unity_editor.request("LoadPointCloud", generate_point_cloud_data(rgb_image, depth_image))
net_manager.spin()

# data = simpublisher.generate_point_cloud_data(rgb_image, depth_image)
# simpublisher.join()
