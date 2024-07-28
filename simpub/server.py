from __future__ import annotations
import abc
from typing import Dict, List, Callable
import zmq
import time
from time import sleep
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import json
import threading
from enum import Enum
import struct
from concurrent.futures import ThreadPoolExecutor, Future

from simpub.simdata import SimScene


class PortSet(int, Enum):
    DISCOVERY = 7720
    SERVICE = 7721
    STREAMING = 7722


class TaskBase(abc.ABC):

    def __init__(self):
        self.running: bool = False

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def execute(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class BroadcastTask(TaskBase):

    def __init__(
        self,
        discovery_message: str,
        host: str = "127.0.0.1",
        mask: str = "255.255.255.0",
        port: int = PortSet.DISCOVERY,
        intervall: float = 1.0,
    ):
        self._port = port
        self.running = True
        self._intervall = intervall
        self._message = discovery_message.encode()
        # calculate broadcast ip
        ip_bin = struct.unpack('!I', socket.inet_aton(host))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton(mask))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        self.broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))

    def execute(self):
        self.running = True
        self.conn = socket.socket(AF_INET, SOCK_DGRAM)
        self.conn.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        while self.running:
            self.conn.sendto(self._message, (self.broadcast_ip, self._port))
            sleep(self._intervall)

    def on_shutdown(self):
        self.running = False


class StreamTask(TaskBase):

    def __init__(
        self,
        context: zmq.Context,
        update_func: Callable[[], Dict],
        host: str = "127.0.0.1",
        port: int = PortSet.STREAMING,
        topic: str = "SceneUpdate",
        fps: int = 45,
    ):
        self._context: zmq.Context = context
        self._update_func = update_func
        self._topic: str = topic
        self.running: bool = False
        self._dt: float = 1 / fps
        self.pub_socket: zmq.Socket = self._context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host}:{port}")

    def execute(self):
        print("Stream task has been started")
        self.running = True
        last = 0.0
        while self.running:
            diff = time.monotonic() - last
            if diff < self._dt:
                time.sleep(self._dt - diff)
            last = time.monotonic()
            msg = {
                "updateData": self._update_func(),
                "time": time.monotonic()
            }
            self.pub_socket.send_string(f"{self._topic}:{json.dumps(msg)}")

    def on_shutdown(self):
        self.pub_socket.close(0)


class SubscribeTask(TaskBase):

    def __init__(
        self,
        context: zmq.Context,
        callback_func: Callable[[], Dict],
        host: str,
        port: int,
        topic: str,
    ):
        self._context: zmq.Context = context
        self._callback_func = callback_func
        self._topic: str = topic
        self.running: bool = False
        self.sub_socket: zmq.Socket = self._context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{host}:{port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def execute(self):
        print("Subscribe task has been started")
        self.running = True
        while self.running:
            message = self.sub_socket.recv_string()
            self._callback_func(message)


class MsgService(TaskBase):

    def __init__(
        self,
        context: zmq.Context,
        host: str = "127.0.0.1",
        port: int = PortSet.SERVICE,
    ):
        self._context: zmq.Context = context
        self._port: int = port
        self.running: bool = False
        self._actions: Dict[str, Callable[[zmq.Socket, str], None]] = {}
        self.reply_socket: zmq.Socket = self._context.socket(zmq.REP)
        self.reply_socket.bind(f"tcp://{host}:{port}")

    def execute(self):
        self.running = True
        while self.running:
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
        self.running = False




class ServerBase(abc.ABC):

    def __init__(self, host: str = "127.0.0.1"):
        self.host: str = host
        self.running: bool = False
        self.executor: ThreadPoolExecutor
        self.futures: List[Future] = []
        self.tasks: List[TaskBase] = []
        self.zmqContext = zmq.Context()
        self.initialize_task()

    @abc.abstractmethod
    def initialize_task(self):
        raise NotImplementedError

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.thread_task)
        self.thread.start()

    def join(self):
        self.thread.join()

    def thread_task(self):
        print("Server Tasks has been started")
        with ThreadPoolExecutor(max_workers=5) as executor:
            self.executor = executor
            for task in self.tasks:
                self.futures.append(executor.submit(task.execute))
        self.running = False

    def shutdown(self):
        print("Trying to shutdown server")
        for task in self.tasks:
            task.shutdown()
        self.thread.join()
        print("All the threads have been stopped")

    def add_task(self, task: TaskBase):
        self.tasks.append(task)


class MsgServer(ServerBase):
    def __init__(self, host: str = "127.0.0.1"):
        super().__init__(host)

    def initialize_task(self):    
        self.tasks: List[TaskBase] = []
        discovery_data = {
            "SERVICE": PortSet.SERVICE,
        }
        time_stamp = str(time.time())
        discovery_message = f"SimPub:{time_stamp}:{json.dumps(discovery_data)}"
        self.broadcast_task = BroadcastTask(discovery_message, self.host)
        self.tasks.append(self.broadcast_task)

        self.msg_service = MsgService(self.zmqContext, self.host)
        self.tasks.append(self.msg_service)


class SimPublisher(ServerBase):

    def __init__(
        self,
        sim_scene: SimScene,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        self.sim_scene = sim_scene
        if no_rendered_objects is None:
            self.no_rendered_objects = []
        else:
            self.no_rendered_objects = no_rendered_objects
        if no_tracked_objects is None:
            self.no_tracked_objects = []
        else:
            self.no_tracked_objects = no_tracked_objects
        super().__init__(host)

    def initialize_task(self):
        self.tasks: List[TaskBase] = []
        discovery_data = {
            "SERVICE": PortSet.SERVICE,
            "STREAMING": PortSet.STREAMING,
        }
        time_stamp = str(time.time())
        discovery_message = f"SimPub:{time_stamp}:{json.dumps(discovery_data)}"
        self.broadcast_task = BroadcastTask(discovery_message, self.host)
        self.add_task(self.broadcast_task)

        self.stream_task = StreamTask(
            self.zmqContext, self.get_update, self.host
        )
        self.add_task(self.stream_task)

        self.msg_service = MsgService(self.zmqContext, self.host)
        self.msg_service.register_action("SCENE", self._on_scene_request)
        self.msg_service.register_action("ASSET", self._on_asset_request)
        self.add_task(self.msg_service)

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
