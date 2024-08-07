import random
import cv2

import json
from simpub.core.simpub_server import ServerBase
import zmq


class RGBDPublisher(ServerBase):

    def __init__(self, remote_ip: str, host: str = "127.0.0.1") -> None:
        super().__init__(host)
        self.request_socket = self.net_manager.zmq_context.socket(zmq.REQ)
        self.request_socket.connect(f"tcp://{remote_ip}:7723")

    def generate_point_cloud_data(self, rgb_image, depth_image):
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
        return {"positions": positions, "colors": colors}

    def send_point_cloud(self, point_data) -> None:
        self.request_socket.send_string(json.dumps(self.point_cloud_data))


rgb_path = '/home/xinkai/project/point-cloud-labeling/rgb.png'
depth_path = '/home/xinkai/project/point-cloud-labeling/depth.png'

rgb_image = cv2.imread(rgb_path)
depth_image = cv2.imread(depth_path)

simpublisher = RGBDPublisher(rgb_image, depth_image)
data = simpublisher.generate_point_cloud_data(rgb_image, depth_image)
simpublisher.join()
