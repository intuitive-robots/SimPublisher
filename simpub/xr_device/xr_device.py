import json
import zmq
import traceback
import time
from typing import Optional
from asyncio import sleep as async_sleep

from ..core.log import logger
from ..core.node_manager import init_xr_node_manager, XRNodeInfo
from ..core.net_component import Subscriber
from ..core.utils import AsyncSocket, send_request_async


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDevice:
    type = "XRDevice"

    def __init__(
        self,
        device_name: str = "UnityNode",
    ) -> None:
        self.manager = init_xr_node_manager()
        self.running = True
        self.connected = False
        self.device_name = device_name
        self.device_id: Optional[str] = None
        self.device_info: Optional[XRNodeInfo] = None
        self.req_socket: AsyncSocket = self.manager.create_socket(zmq.REQ)
        self.sub_list: list[Subscriber] = []
        self.sub_list.append(Subscriber("ConsoleLogger", self.print_log))
        self.manager.submit_asyncio_task(self.checking_connection)

    def wait_for_connection(self):
        """
        Wait for the connection to the XR device.
        This method blocks until the connection is established.
        """
        while not self.connected:
            time.sleep(0.1)

    async def checking_connection(self):
        logger.info(f"checking the connection to {self.device_name}")
        while self.running:
            for entry in self.manager.xr_nodes.values():
                node_info = entry.info
                if node_info is None:
                    continue
                if node_info["name"] != self.device_name:
                    continue
                if node_info["nodeID"] == self.device_id:
                    continue
                if self.device_info is not None:
                    self.disconnect()
                self.device_info = node_info
                self.device_id = node_info["nodeID"]
                self.connected = True
                self.subscribe_to_client(node_info)
                self.print_node_info(node_info)
            await async_sleep(0.5)
        # if self.device_info is None:
        #     return
        # self.connect_to_client(self.device_info)

    def subscribe_to_client(self, info: XRNodeInfo):
        try:
            self.req_socket.connect(f"tcp://{info['ip']}:{info['port']}")
            for sub in self.sub_list:
                if sub.topic_name not in info["topicDict"]:
                    continue
                sub.start_connection(
                    f"tcp://{info['ip']}:{info['topicDict'][sub.topic_name]}"
                )
                logger.info(
                    f"Subscribed to {sub.topic_name} "
                    f"on {info['ip']}:{info['port']}"
                )
            logger.remote_log(
                f"{self.type} Connected to {self.device_name} "
                f"at {info['ip']}:{info['port']}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to {self.device_name} at "
                f"{info['ip']}:{info['port']}: {e}"
            )
            traceback.print_exc()
            return

    def request(self, service_name: str, request: str) -> str:
        if self.device_info is None:
            logger.error(f"Device {self.device_name} is not connected")
            return ""
        if service_name not in self.device_info["serviceList"]:
            logger.error(f'"{service_name}" Service is not available')
            return ""
        messages = [
            service_name.encode(),
            request.encode(),
        ]
        future = self.manager.submit_asyncio_task(
            send_request_async, messages, self.req_socket
        )
        if future is None:
            logger.error("Future is None")
            return ""
        try:
            result = future.result()
            return result
        except Exception as e:
            logger.error(f"Error occurred when waiting for a response: {e}")
            return ""

    def disconnect(self):
        if self.device_info is None:
            logger.error(
                f"Device {self.device_name} is not "
                "connected and it cannot be disconnected"
            )
            return
        self.req_socket.disconnect(
            f"tcp://{self.device_info['ip']}:{self.device_info['port']}"
        )

    def print_log(self, log: str):
        logger.remote_log(f"{self.type} Log: {log}")

    def get_controller_data(self) -> InputData:
        raise NotImplementedError

    def print_node_info(self, node_info: XRNodeInfo):
        # print the node info with service and topics
        logger.info(f"Node Info for {node_info['name']}:")
        logger.info(f"  Node ID: {node_info['nodeID']}")
        logger.info(f"  IP: {node_info['ip']}")
        logger.info(f"  Port: {node_info['port']}")
        logger.info(f"  Services: {node_info['serviceList']}")
        logger.info(f"  Topics: {node_info['topicDict']}")
