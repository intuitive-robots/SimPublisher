import enum
from typing import List, Dict
from typing import NewType, Callable, TypedDict, Union
import asyncio
from asyncio import sleep as asycnc_sleep
import threading
import zmq
import zmq.asyncio
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
import time
import json
import uuid
from .log import logger
from json import dumps
import zmq
import abc

IPAddress = NewType("IPAddress", str)
TopicName = NewType("TopicName", str)
ServiceName = NewType("ServiceName", str)


class ServerPort(int, enum.Enum):
    # ServerPort and ClientPort need using .value to get the port number
    # which is not supposed to be used in this way
    DISCOVERY = 7720
    SERVICE = 7721
    TOPIC = 7722


class ClientPort(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7723
    TOPIC = 7724


class HostInfo(TypedDict):
    name: str
    ip: IPAddress
    topics: List[TopicName]
    services: List[ServiceName]


class Communicator(abc.ABC):

    def __init__(self):
        self.running: bool = False
        self.manager: NetManager = NetManager.manager
        self.host_ip: str = self.manager.local_info["ip"]
        self.host_name: str = self.manager.local_info["host"]

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(Communicator):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.socket = self.manager.pub_socket
        self.manager.register_local_topic(self.topic)
        logger.info(f"Publisher for topic \"{self.topic}\" is ready")

    def publish(self, data: Dict):
        msg = f"{self.topic}:{dumps(data)}"
        self.socket.send_string(msg)

    def publish_string(self, string: str):
        self.socket.send_string(f"{self.topic}:{string}")

    def on_shutdown(self):
        super().on_shutdown()


class Streamer(Publisher):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], Dict],
        fps: int = 45,
    ):
        super().__init__(topic)
        self.running = False
        self.dt: float = 1 / fps
        self.update_func = update_func
        self.manager.submit_task(self.update_loop)

    async def update_loop(self):
        self.running = True
        last = 0.0
        while self.running:
            diff = time.monotonic() - last
            if diff < self.dt:
                await asycnc_sleep(self.dt - diff)
            last = time.monotonic()
            msg = {
                "updateData": self.update_func(),
                "time": last,
            }
            self.socket.send_string(f"{self.topic}:{dumps(msg)}")


ResponseType = Union[str, bytes, Dict]


class Service(Communicator):

    def __init__(
        self,
        service_name: str,
        callback: Callable[[str], ResponseType],
        respnse_type: ResponseType,
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.callback_func = callback
        if respnse_type == str:
            self.sender = self.send_string
        elif respnse_type == bytes:
            self.sender = self.send_bytes
        elif respnse_type == Dict:
            self.sender = self.send_dict
        else:
            raise ValueError("Invalid response type")
        self.socket = self.manager.service_socket
        # register service
        self.manager.local_info["services"].append(service_name)
        self.manager.service_list[service_name] = self
        logger.info(f"\"{self.service_name}\" Service is ready")

    async def callback(self, msg: str):
        result = await asyncio.wait_for(
            asyncio.to_thread(self.callback_func, msg),
            timeout=5
        )
        await self.sender(result)

    async def send_string(self, string: str):
        await self.socket.send_string(string)

    async def send_bytes(self, data: bytes):
        await self.socket.send(data)

    async def send_dict(self, data: Dict):
        await self.socket.send_string(dumps(data))

    def on_shutdown(self):
        return super().on_shutdown()


class NetManager:

    manager = None

    def __init__(
        self,
        host_ip: IPAddress = "127.0.0.1",
        host_name: str = "SimPub"
    ) -> None:
        NetManager.manager = self
        self._initialized = True
        self.zmq_context = zmq.asyncio.Context()
        # subscriber
        self.sub_socket_dict: Dict[IPAddress, zmq.Socket] = {}
        # publisher
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:{ServerPort.TOPIC.value}")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:{ServerPort.SERVICE.value}")
        self.service_list: Dict[str, Service] = {}
        # message for broadcasting
        self.local_info = HostInfo()
        self.local_info["host"] = host_name
        self.local_info["ip"] = host_ip
        self.local_info["topics"] = []
        self.local_info["services"] = []
        # host info
        self.clients_info: Dict[str, HostInfo] = {}
        # setting up thread pool
        self.running: bool = True
        self.loop: asyncio.AbstractEventLoop = None
        self.start_server_thread()

    def start_server_thread(self) -> None:
        """
        Start a thread for service.

        Args:
            block (bool, optional): main thread stop running and
            wait for server thread. Defaults to False.
        """
        self.server_thread = threading.Thread(target=self.start_event_loop)
        self.server_thread.daemon = True
        self.server_thread.start()
        while self.loop is None:
            time.sleep(0.01)

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # default task for client registration
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)
        self.loop.run_forever()

    def submit_task(self, task: Callable, *args) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def stop_server(self):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.loop.stop(), self.loop)
        self.server_thread.join()

    def join(self):
        self.server_thread.join()

    async def service_loop(self):
        # try:
        logger.info("The service is running...")
        # default service for client registration
        self.register_service = Service(
            "Register", self.register_client_callback, str
        )
        self.server_timestamp_service = Service(
            "GetServerTimestamp", self.get_server_timestamp_callback, str
        )
        while self.running:
            message = await self.service_socket.recv_string()
            if ":" not in message:
                logger.error(f"Invalid message with no spliter \":\"")
                await self.service_socket.send_string("Invalid message")
                continue        
            service, request = message.split(":", 1)
            # the zmq service socket is blocked and only run one at a time
            if service in self.service_list.keys():
                try:
                    await self.service_list[service].callback(request)
                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout: callback function took too long to execute"
                    )
                    await self.service_socket.send_string("Timeout")
                except Exception as e:
                    logger.error(f"Error: {e}")
            await asycnc_sleep(0.01)

    async def broadcast_loop(self):
        logger.info("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        _id = str(uuid.uuid4())
        # calculate broadcast ip
        local_info = self.local_info
        ip_bin = struct.unpack('!I', socket.inet_aton(local_info["ip"]))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))
        while self.running:
            msg = f"SimPub:{_id}:{json.dumps(local_info)}"
            _socket.sendto(
                msg.encode(), (broadcast_ip, ServerPort.DISCOVERY.value)
            )
            await asycnc_sleep(0.1)
        logger.info("Broadcasting has been stopped")

    def register_client_callback(self, msg: str) -> str:
        # NOTE: something woring with sending message, but it solved somehow
        client_info: HostInfo = json.loads(msg)
        client_name = client_info["name"]
        # NOTE: the client info may be updated so the reference cannot be used
        # NOTE: TypeDict is somehow block if the key is not in the dict
        self.clients_info[client_name] = client_info
        logger.info(f"Host \"{client_name}\" has been registered")
        return "The info has been registered"

    def get_server_timestamp_callback(self, msg: str) -> str:
        return str(time.monotonic())

    def register_local_topic(self, topic: TopicName):
        if topic in self.local_info["topics"]:
            logger.warning(f"Host {topic} is already registered")
        self.local_info["topics"].append(topic)

    def shutdown(self):
        logger.info("Shutting down the server")
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        self.running = False
        logger.info("Server has been shut down")


def init_net_manager(host: str):
    if NetManager.manager is not None:
        return NetManager.manager
    return NetManager(host)
