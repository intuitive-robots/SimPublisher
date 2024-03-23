from abc import ABCMeta
import asyncio
from asyncio import sleep as async_sleep
import zmq
from zmq.asyncio import Context as AsyncContext
from zmq.asyncio import Socket as AsyncSocket
from typing import List, Tuple

from ..base import ServerBase

BROADCAST_PORT = 5554
REQ_PORT = 5555
TOPIC_PORT = 5556
UDP_PORT = 5557

class ZMQServer(ServerBase):
    """
    A server class for running a service on a simulation environment by ZMQ.
    """
    def __init__(self) -> None:
        super().__init__()
        
        self._context: AsyncContext = None
        self._broadcast_socket: AsyncSocket = None
        self._request_socket: AsyncSocket = None
        self._topic_socket: AsyncSocket = None
        self._udp_socket: AsyncSocket = None

        self._client_list: List[Tuple[str, str]] = list()

    async def handler(self):
        """
        The main task handler of loop in server thread.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """
        self._loop = asyncio.get_event_loop()
        self._request_socket = self._context.socket(zmq.REQ)
        self._response_socket = self._context.socket(zmq.REP)
        self._message_socket = self._context.socket(zmq.PUB)
        self._udp_socket = self._context.socket(zmq.RADIO)
        _, pending = await asyncio.wait(
            (self._request_handler(),
             self._broadcast_handler(),),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    def _server_task(self) -> None:
        self._context = AsyncContext()
        asyncio.run(self.handler())
        self._context.term()
        print("server task finished")

    async def _broadcast_handler(self):
        """
        The coroutine task for broadcasting messages on loop.
        """
        self._broadcast_socket = self._context.socket(zmq.PUB)
        self._broadcast_socket.bind(f"tcp://{self._host}:{BROADCAST_PORT}")
        while self.is_active:
            await self._broadcast_socket.send_string(f"tcp://{self._host}:{REQ_PORT}")
            await async_sleep(5)
        self._broadcast_socket.close()

    async def _request_handler(self):
        """
        The coroutine task for handling request on loop.
        """
        self._request_socket.bind(f"tcp://{self._host}:{REQ_PORT}")
        while self.is_active:
            msg = await self._request_socket.recv_string()
            await self._request_socket.send_string(f"Received: {msg}")
        self._request_socket.close()

    def send_request(self, msg: str) -> str:
        """
        Send a request message to the server.

        Args:
            msg (str): message to be sent.
        Returns:
            str: response message from the server.
        """
        if self._response_socket is None:
            return
        self._loop.create_task(self._response_socket.send_string(msg))

        return self._loop.create_task(self._response_socket.recv_string())
    

    def register_client(self, client_id: str, client_address: str) -> None:
        """
        Register a client to the server.

        Args:
            client_id (str): client id.
            client_address (str): client address.
        """
        self._client_list.append((client_id, client_address))