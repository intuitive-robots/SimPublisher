# from __future__ import annotations

# import pyzlc
# from pyzlc.nodes.loop_manager import LanComLoopManager

# import abc
# import time
# import traceback
# from asyncio import sleep as async_sleep
# from json import dumps
# from typing import Callable, Dict, Optional, Union

# # import zmq

# # from .log import logger
# # from .node_manager import init_xr_node_manager
# # from .utils import AsyncSocket


# # class NetComponent(abc.ABC):
# #     def __init__(self):
# #         self.manager = init_xr_node_manager()
# #         self.running: bool = False

# #     def shutdown(self) -> None:
# #         self.running = False
# #         self.on_shutdown()

# #     @abc.abstractmethod
# #     def on_shutdown(self):
# #         raise NotImplementedError


# # class Publisher(NetComponent):
# #     def __init__(self, topic_name: str):
# #         super().__init__()
# #         self.topic_name = topic_name
# #         self.socket = self.manager.create_socket(zmq.PUB)
# #         self.socket.bind(f"tcp://{self.manager.host_ip}:0")

# #     def publish_bytes(self, msg: bytes) -> None:
# #         self.manager.submit_asyncio_task(self.send_bytes_async, msg)

# #     def publish_dict(self, msg: Dict) -> None:
# #         self.publish_string(dumps(msg))

# #     def publish_string(self, msg: str) -> None:
# #         self.manager.submit_asyncio_task(self.send_bytes_async, msg.encode())

# #     async def send_bytes_async(self, msg: bytes) -> None:
# #         await self.socket.send(msg)

# #     def on_shutdown(self):
# #         pass


# class Streamer(pyzlc.Publisher):
#     def __init__(
#         self,
#         topic_name: str,
#         update_func: Callable[[], Dict],
#         fps: int,
#         start_streaming: bool = False,
#     ):
#         super().__init__(topic_name)
#         self.running = False
#         self.dt: float = 1 / fps
#         self.update_func = update_func
#         if start_streaming:
#             self.start_streaming()

#     def start_streaming(self):
#         assert LanComLoopManager.instance is not None, "pyzlc is not initialized"
#         LanComLoopManager.instance.submit_loop_task(self.update_loop())

#     # def generate_byte_msg(self) -> bytes:
#     #     update_msg = self.update_func()
#     #     if isinstance(update_msg, str):
#     #         return update_msg.encode("utf-8")
#     #     elif isinstance(update_msg, bytes):
#     #         return update_msg
#     #     elif isinstance(update_msg, dict):
#     #         return dumps(update_msg).encode("utf-8")
#     #     raise ValueError("Update function should return str, bytes or dict")

#     async def update_loop(self):
#         self.running = True
#         last = 0.0
#         pyzlc.info(f"Topic {self.name} starts streaming")
#         while self.running:
#             try:
#                 diff = time.monotonic() - last
#                 if diff < self.dt:
#                     await async_sleep(self.dt - diff)
#                 last = time.monotonic()
#                 await self.publish(self.update_func())
#             except Exception as e:
#                 pyzlc.error(f"Error when streaming {self.name}: {e}")
#                 traceback.print_exc()
#         pyzlc.info(f"Streamer for topic {self.name} is stopped")


# # class ByteStreamer(Streamer):
# #     def __init__(
# #         self,
# #         topic: str,
# #         update_func: Callable[[], bytes],
# #         fps: int,
# #     ):
# #         super().__init__(topic, update_func, fps)
# #         self.update_func: Callable[[], bytes]

# #     def generate_byte_msg(self) -> bytes:
# #         return self.update_func()


# # class Subscriber(NetComponent):
# #     def __init__(self, topic_name: str, callback: Callable[[str], None]):
# #         super().__init__()
# #         self.sub_socket: AsyncSocket = self.manager.create_socket(zmq.SUB)
# #         self.topic_name = topic_name
# #         self.callback = callback
# #         self.remote_addr: Optional[str] = None
# #         self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

# #     def start_connection(self, addr: str) -> None:
# #         """Changes the connection to a new IP address."""
# #         # if self.remote_addr is not None:
# #         #     logger.info(f"Disconnecting from {self.remote_addr}")
# #         #     self.sub_socket.disconnect(f"tcp://{self.remote_addr}")
# #         self.sub_socket.connect(addr)
# #         self.remote_addr = addr
# #         self.running = True
# #         self.manager.submit_asyncio_task(self.listen)
# #         logger.info(
# #             f"Subscriber for topic '{self.topic_name}' started "
# #             f"listening on {addr}"
# #         )

# #     async def listen(self) -> None:
# #         """Listens for incoming messages on the subscribed topic."""
# #         while self.running:
# #             try:
# #                 msg = await self.sub_socket.recv_string()
# #                 self.callback(msg)
# #             except Exception as e:
# #                 logger.error(
# #                     "Error in callback function of subscriber"
# #                     f" for topic '{self.topic_name}': {e}"
# #                 )
# #                 traceback.print_exc()

# #     def on_shutdown(self) -> None:
# #         self.running = False
# #         self.sub_socket.close()
