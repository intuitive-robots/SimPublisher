from simpub.core.node_manager import init_xr_node_manager
from simpub.xr_device.xr_device import XRDevice

if __name__ == "__main__":
    node_manager = init_xr_node_manager("192.168.0.134")
    node_manager.start_discover_node_loop()
    device = XRDevice()
    node_manager.spin()
