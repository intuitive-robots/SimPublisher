import json
import zmq
import traceback
from typing import Optional
from asyncio import sleep as async_sleep

from ..core.log import logger
from ..core.node_manager import init_xr_node_manager, XRNodeInfo
from ..core.net_component import Subscriber

# from ..core.utils import TopicName


# from ..core.net_component import Subscriber
from ..core.utils import AsyncSocket


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

    async def checking_connection(self):
        logger.info(f"checking the connection to {self.device_name}")
        while self.running:
            for node_info in self.manager.xr_nodes_info.values():
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

    def request(self, service: str, request: str) -> str:
        future = self.manager.submit_asyncio_task(
            self.request_async, service, request
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

    async def request_async(self, service: str, req: str) -> str:
        if self.device_info is None:
            logger.error(f"Device {self.device_name} is not connected")
            return ""
        if service not in self.device_info["serviceList"]:
            logger.error(f'"{service}" Service is not available')
            return ""
        await self.req_socket.send_string(f"{service}:{req}")
        return await self.req_socket.recv_string()

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
