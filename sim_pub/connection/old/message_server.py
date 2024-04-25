import asyncio
from asyncio import sleep as async_sleep
import threading
import zmq
from zmq.asyncio import Context as AsyncContext
from zmq.asyncio import Socket as AsyncSocket

BROADCAST_PORT = 5554
REQ_PORT = 5555
TOPIC_PORT = 5556
UDP_PORT = 5557

class MessageServer:
    """
    A server class for running a service on a simulation environment by ZMQ.
    """
    def __init__(self, host: str) -> None:
        super().__init__()
        self._server_thread: threading.Thread = None
        self._loop: asyncio.AbstractEventLoop = None

        self._host = host
        self._is_active: bool = False
        
        self._context: AsyncContext = None
        self._broadcast_socket: AsyncSocket = None
        self._service_socket: AsyncSocket = None
        self._stream_socket: AsyncSocket = None
        self._msg_pusher_socket: AsyncSocket = None
        self._msg_receiver_socket: AsyncSocket = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    def start_server_thread(self, block = False) -> None:
        """
        Start a thread for service.

        Args:
            block (bool, optional): main thread stop running and 
            wait for server thread. Defaults to False.
        """
        self._server_thread = threading.Thread(target=self._server_task)
        self._server_thread.daemon = True
        self._server_thread.start()
        if block:
            self._server_thread.join()

    async def handler(self):
        """
        The main task handler of loop in server thread.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """
        self._loop = asyncio.get_event_loop()
        self._service_socket = self._context.socket(zmq.REQ)
        self._message_socket = self._context.socket(zmq.PUB)
        self._udp_socket = self._context.socket(zmq.RADIO)
        # self._response_socket = self._context.socket(zmq.REP)
        _, pending = await asyncio.wait(
            (self._service_handler(),
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

    async def _service_handler(self):
        """
        The coroutine task for handling request on loop.
        """
        self._service_socket.bind(f"tcp://{self._host}:{REQ_PORT}")
        while self.is_active:
            msg = await self._service_socket.recv_string()
            await self._service_socket.send_string(f"Received: {msg}")
        self._service_socket.close()

