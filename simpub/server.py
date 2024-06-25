from __future__ import annotations
import abc
from typing import Dict, List, Callable
import zmq
import time
from time import sleep
from socket import socket, AF_INET, SOCK_DGRAM
from socket import SOL_SOCKET, SO_BROADCAST
import json
import threading
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future

from simpub.simdata import SimScene


class PortSet(str, Enum):
    DISCOVERY = 5520
    SERVICE = 5521
    STREAMING = 5522


class TaskBase(abc.ABC):

    def __init__(self):
        self._running: bool = False

    def shutdown(self):
        self._running = False
        self.on_shutdown()

    @abc.abstractmethod
    def execute(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class DiscoveryTask(TaskBase):

    def __init__(
        self,
        discovery_message: str,
        port: int = PortSet.DISCOVERY,
        intervall: int = 2
    ):
        self._port = port
        self._running = True
        self._intervall = intervall
        self._message = discovery_message.encode()

    def execute(self):
        print("Discovery Task has been started")
        self._running = True
        self.conn = socket(AF_INET, SOCK_DGRAM)  # create UDP socket
        self.conn.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        while self._running:
            self.conn.sendto(self._message, ("255.255.255.255", self._port))
            sleep(self._intervall)

    def on_shutdown(self):
        self._running = False


class ServerBase(abc.ABC):

    def __init__(self):
        self._running: bool = False
        self.executor: ThreadPoolExecutor
        self.futures: List[Future] = []
        self.tasks: List[TaskBase] = []
        self.zmqContext = zmq.Context()
        self.initialize_task()

    @abc.abstractmethod
    def initialize_task(self):
        raise NotImplementedError

    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self.thread_task)
        self.thread.start()

    def thread_task(self):
        print("Server Tasks has been started")
        with ThreadPoolExecutor(max_workers=5) as executor:
            self.executor = executor
            for task in self.tasks:
                self.futures.append(executor.submit(task.execute))
        self._running = False

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError


class StreamTask(TaskBase):

    def __init__(
        self,
        context: zmq.Context,
        update_func: Callable[[], Dict],
        port: int = PortSet.STREAMING,
        topic: str = "simulation_update",
        fps: int = 30,
    ):
        self._context: zmq.Context = context
        self._update_func = update_func
        self._port: int = port
        self._running: bool = False
        self._dt: float = 1 / fps

    def execute(self):
        print("Stream Task has been started")
        self._running = True
        self.pub_socket: zmq.Socket = self._context.socket(zmq.PUB)
        last = 0.0
        while self._running:
            diff = time.monotonic() - last
            if diff < self._dt:
                time.sleep(self._dt - diff)
            last = time.monotonic()
            msg = {
                "data": self._update_func(),
                "time": time.monotonic()
            }
            self.pub_socket.send_string(json.dumps(msg))

    def on_shutdown(self):
        self.pub_socket.close(0)


class MsgService(TaskBase):

    def __init__(
        self,
        context: zmq.Context,
        port: int = PortSet.SERVICE,
    ):
        self._context: zmq.Context = context
        self._port: int = port
        self._running: bool = False
        self._actions: Dict[str, Callable[[zmq.Socket, str], None]] = {}

    def execute(self):
        self.reply_socket: zmq.Socket = self._context.socket(zmq.REP)
        self.reply_socket.bind(f"tcp://*:{self._port}")
        while self._running:
            message = self.reply_socket.recv().decode()
            tag, *args = message.split(":", 1)
            if tag == "END":
                continue
            if tag not in self._actions:
                print(f"Invalid request {tag}")
                self.reply_socket.send_string("INVALID")
                continue
            self._actions[tag](self.reply_socket, args[0] if args else "")

    def register_action(
        self,
        tag: str,
        action: Callable[[zmq.Socket, str], None]
    ):
        if tag in self._actions:
            raise RuntimeError("Action was already registred", tag)
        self._actions[tag] = action

    def on_shutdown(self):
        self._running = False


class SimPublisher(ServerBase):

    def __init__(self, sim_scene: SimScene) -> None:
        self.sim_scene = sim_scene
        super().__init__()

    def initialize_task(self):
        print("Initializing tasks")
        self.tasks: List[TaskBase] = []
        discovery_data = {
            "SERVICE": PortSet.SERVICE,
            "STREAMING": PortSet.STREAMING,
        }
        discovery_message = f"SimPub:{json.dumps(discovery_data)}"
        self.discovery_task = DiscoveryTask(discovery_message)
        # self.tasks.append(self.discovery_task)

        self.stream_task = StreamTask(self.zmqContext, self.get_update)
        self.tasks.append(self.stream_task)

        self.msg_service = MsgService(self.zmqContext)
        self.msg_service.register_action("SCENE", self._on_scene_request)
        self.msg_service.register_action("ASSET", self._on_asset_request)
        self.tasks.append(self.msg_service)

    def _on_scene_request(self, socket: zmq.Socket, tag: str):
        socket.send_string(self.sim_scene.to_string())

    def _on_asset_request(self, socket: zmq.Socket, tag: str):
        if tag not in self.sim_scene.raw_data:
            print("Received invalid data request")
            return
        socket.send(self.sim_scene.raw_data[tag])

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
