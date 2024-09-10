from simpub.core.simpub_server import init_net_manager
from simpub.xr_device.xr_device import XRDevice
from simpub.core.log import logger

import time
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=str, default="UnityClient")
    parser.add_argument("--new", type=str, default="UnityClient")
    args = parser.parse_args()

    net_manager = init_net_manager("127.0.0.1")
    xr_device = XRDevice(args.old)
    while not xr_device.connected:
        time.sleep(0.01)
    result = xr_device.request("ChangeHostName", args.new)
    logger.info(result)
