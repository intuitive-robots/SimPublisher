import random
import cv2

import json
from simpub.core.net_manager import init_net_manager
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


def send_point_cloud(self, point_data) -> None:
    self.request_socket.send_string(json.dumps(self.point_cloud_data))


rgb_path = '/home/xinkai/project/point-cloud-labeling/rgb.png'
depth_path = '/home/xinkai/project/point-cloud-labeling/depth.png'

rgb_image = cv2.imread(rgb_path)
depth_image = cv2.imread(depth_path)

# simpublisher = RGBDPublisher(rgb_image, depth_image)
net_manager = init_net_manager("127.0.0.1")
unity_editor = XRDevice()
# print(generate_point_cloud_data(rgb_image, depth_image))
while unity_editor.connected is False:
    pass
unity_editor.request("LoadPointCloud", generate_point_cloud_data(rgb_image, depth_image))
net_manager.join()

# data = simpublisher.generate_point_cloud_data(rgb_image, depth_image)
# simpublisher.join()
