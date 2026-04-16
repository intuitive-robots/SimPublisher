import json
import time
import traceback
from asyncio import sleep as async_sleep
from typing import Optional

import pyzlc
from pyzlc.utils.node_info import NodeInfo as XRNodeInfo

from ..core.utils import ZLC_GROUP_NAME, print_node_info


class XRDevice:
    type = "XRDevice"

    def __init__(
        self,
        device_type: str,
        device_name: str = "UnityNode",
    ) -> None:
        self.manager = pyzlc.get_node(group_name=ZLC_GROUP_NAME)
        self.running = True
        self.connected = False
        self.device_name = device_name
        self.device_full_name = f"IRIS/Device/{device_type}/{device_name}"
        self.device_id: Optional[str] = None
        self.device_info: Optional[XRNodeInfo] = None
        pyzlc.submit_loop_task(self.checking_connection(), group_name=ZLC_GROUP_NAME)

    def wait_for_connection(self):
        """
        Wait for the connection to the XR device.
        This method blocks until the connection is established.
        """
        while not self.connected:
            time.sleep(0.1)

    async def checking_connection(self):
        pyzlc.info(f"checking the connection to {self.device_full_name}")
        while self.running:
            for node_info in pyzlc.get_nodes_info(group_name=ZLC_GROUP_NAME):
                if node_info["name"] != self.device_full_name:
                    continue
                if node_info["nodeID"] == self.device_id:
                    continue
                if self.device_info is not None:
                    self.disconnect()
                self.device_info = node_info
                self.device_id = node_info["nodeID"]
                self.connected = True
            await async_sleep(0.5)
        # if self.device_info is None:
        #     return
        # self.connect_to_client(self.device_info)

    # def subscribe_to_client(self, info: XRNodeInfo):
    #     try:
    #         self.req_socket.connect(f"tcp://{info['ip']}:{info['port']}")
    #         for sub in self.sub_list:
    #             if sub.topic_name not in info["topicDict"]:
    #                 continue
    #             sub.start_connection(
    #                 f"tcp://{info['ip']}:{info['topicDict'][sub.topic_name]}"
    #             )
    #             pyzlc.info(
    #                 f"Subscribed to {sub.topic_name} "
    #                 f"on {info['ip']}:{info['port']}"
    #             )
    #         pyzlc.remote_log(
    #             f"{self.type} Connected to {self.device_name} "
    #             f"at {info['ip']}:{info['port']}"
    #         )
    #     except Exception as e:
    #         pyzlc.error(
    #             f"Failed to connect to {self.device_name} at "
    #             f"{info['ip']}:{info['port']}: {e}"
    #         )
    #         traceback.print_exc()
    #         return

    # def request(self, service_name: str, request: str) -> str:
    #     if self.device_info is None:
    #         pyzlc.error(f"Device {self.device_name} is not connected")
    #         return ""
    #     if service_name not in self.device_info["serviceList"]:
    #         pyzlc.error(f'"{service_name}" Service is not available')
    #         return ""
    #     messages = [
    #         service_name.encode(),
    #         request.encode(),
    #     ]
    #     future = self.manager.submit_asyncio_task(
    #         send_request_async, messages, self.req_socket
    #     )
    #     if future is None:
    #         pyzlc.error("Future is None")
    #         return ""
    #     try:
    #         result = future.result()
    #         return result
    #     except Exception as e:
    #         pyzlc.error(f"Error occurred when waiting for a response: {e}")
    #         return ""

    def disconnect(self):
        pyzlc.info(f"Disconnecting from {self.device_name}")
        self.connected = False