from simpub.core.node_manager import init_xr_node_manager
from simpub.xr_device.meta_quest3 import MetaQuest3
import time

if __name__ == "__main__":

    net_manager = init_xr_node_manager("192.168.0.134")
    net_manager.start_discover_node_loop()
    mq3 = MetaQuest3("UnityNode")  # You can change the name by using simpubweb
    try:
        while True:
            if mq3.device_info is not None:
                print(f"Device Info: {mq3.device_info}")
            input_data = mq3.get_controller_data()
            if input_data:
                print(f"Received input data: {input_data}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # data = simpublisher.generate_point_cloud_data(rgb_image, depth_image)
    # simpublisher.join()
