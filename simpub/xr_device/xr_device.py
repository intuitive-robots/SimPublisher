import json
import zmq
import zmq.asyncio
from typing import Optional, Dict, Callable
from asyncio import sleep as async_sleep

from ..core.log import logger
from ..core.net_manager import NodeManager, NodeInfo
from ..core.net_manager import TopicName
from ..core.utils import AsyncSocket


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDevice:
    type = "XRDevice"

    def __init__(
        self,
        device_name: str = "UnityClient",
    ) -> None:
        if NodeManager.manager is None:
            raise Exception("NodeManager is not initialized")
        self.manager: NodeManager = NodeManager.manager
        self.connected = False
        self.device_name = device_name
        self.device_info: Optional[NodeInfo] = None
        self.req_socket: AsyncSocket = self.manager.create_socket(zmq.REQ)
        self.sub_socket: AsyncSocket = self.manager.create_socket(zmq.SUB)
        # subscriber
        self.sub_topic_callback: Dict[TopicName, Callable] = {}
        self.register_topic_callback(f"{device_name}/Log", self.print_log)
        self.manager.submit_task(self.wait_for_connection)

    async def wait_for_connection(self):
        logger.info(f"Waiting for connection to {self.device_name}")
        while not self.connected:
            device_info = self.manager.nodes_info_manager.get_node_info(
                self.device_name
            )
            if device_info is not None:
                self.device_info = device_info
                self.connected = True
                logger.info(f"Connected to {self.device_name}")
                break
            await async_sleep(0.01)
        if self.device_info is None:
            return
        self.connect_to_client(self.device_info)

    def connect_to_client(self, info: NodeInfo):
        self.req_socket.connect(
            f"tcp://{info['addr']['ip']}:{info['servicePort']}"
        )
        self.sub_socket.connect(
            f"tcp://{info['addr']['ip']}:{info['topicPort']}"
        )
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.manager.submit_task(self.subscribe_loop)

    def register_topic_callback(self, topic: TopicName, callback: Callable):
        self.sub_topic_callback[topic] = callback

    def request(self, service: str, req: str) -> str:
        future = self.manager.submit_task(
            self.request_async, service, req
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
            logger.error(f"\"{service}\" Service is not available")
            return ""
        await self.req_socket.send_string(f"{service}:{req}")
        return await self.req_socket.recv_string()

    async def subscribe_loop(self):
        try:
            while self.connected:
                message = await self.sub_socket.recv_string()
                topic, msg = message.split("|", 1)
                if topic in self.sub_topic_callback:
                    self.sub_topic_callback[topic](msg)
                await async_sleep(0.01)
        except Exception as e:
            logger.error(
                f"{e} from subscribe loop in device {self.device_name}"
            )

    def print_log(self, log: str):
        logger.remote_log(f"{self.type} Log: {log}")

    def get_input_data(self) -> InputData:
        raise NotImplementedError
