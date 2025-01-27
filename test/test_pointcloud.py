import zmq
import time
import numpy as np


class PointCloudPublisher:

    def __init__(self, ip_addr: str, topic_name: str = "PointCloud"):
        self.topic_name = topic_name
        self.pub_socket = zmq.Context().socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{ip_addr}:7721")
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)

    def publish_bytes(self, data: bytes) -> None:
        msg = b''.join([f"{self.topic_name}".encode(), b"|", data])
        self.pub_socket.send(msg)


def generate_point() -> bytes:
    point = np.random.rand(7)
    point[-1] = 0.005
    return point.astype(np.float32).tobytes()


def generate_point_cloud(point_num: int = 10000) -> bytes:
    point_list = []
    for _ in range(point_num):
        point_list.append(generate_point())
    return b"".join(point_list)


if __name__ == "__main__":

    publisher = PointCloudPublisher("192.168.0.134")
    try:
        while True:
            point_cloud = generate_point_cloud()
            publisher.publish_bytes(point_cloud)
            time.sleep(0.01)
    except KeyboardInterrupt:
        publisher.pub_socket.close()