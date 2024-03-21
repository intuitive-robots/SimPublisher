from websockets import server
import asyncio
from asyncio import sleep as async_sleep
import threading
import json
import abc
from time import sleep as time_sleep
import zmq
from zmq.asyncio import Context as AsyncContext
from zmq.asyncio import Socket as AsyncSocket

from .msg import ObjData, MsgPack

BROADCAST_PORT = 5554
REQ_PORT = 5555
TOPIC_PORT = 5556
UDP_PORT = 5557

class SPBaseServer:
    """
    A base class for all server running on a simulation environment.

    The server receives and sends messages to client by websockets.

    Args:
        host (str): the ip address of service
        port (str): the port of service
    """ 
    def __init__(
        self, 
        host: str, 
        port: str
    ) -> None:

        self._context: AsyncContext = None
        self._broadcast_socket: AsyncSocket = None
        self._request_socket: AsyncSocket = None
        self._response_socket: AsyncSocket = None
        self._udp_socket: AsyncSocket = None

        self._loop: asyncio.AbstractEventLoop = None

        self.host = host
        self.port = port
        self.on_start: bool = False
        self.socket_list: list[AsyncSocket] = list()
        
    def start_server_thread(self, block = False) -> None:
        """
        Start a thread for service.

        Args:
            block (bool, optional): main thread stop running and 
            wait for server thread. Defaults to False.
        """
        self.server_thread = threading.Thread(target=self._server_task)
        self.server_thread.daemon = True
        self.server_thread.start()
        if block:
            self.server_thread.join()

    def wait_for_connection(self) -> None:
        while not self.on_connect:
            time_sleep(0.2)
        return 

    @abc.abstractmethod
    async def process_message(self, msg:str):
        """
        Processing message when receives new messages from clients.

        Args:
            msg (str): message from client.
        """
        raise NotImplementedError

    def create_handler(self, ws: server.WebSocketServerProtocol) -> list[asyncio.Task]:
        """
        Create new tasks such as receiving and stream when start service.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.

        Returns:
            list[asyncio.Task]: the list of new tasks
        """        
        return [
            asyncio.create_task(self._broadcast_handler()),
            asyncio.create_task(self._request_handler()),
        ]

    async def receive_handler(self, ws: server.WebSocketServerProtocol):
        """
        Default messages receiving task handler.

        Args:
        ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """        
        async for msg in ws:
            await self.process_message(msg)

    async def handler(self, ws: server.WebSocketServerProtocol):
        """
        The main task handler of loop in server thread.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """
        self._loop = asyncio.get_event_loop()
        await self.on_connect(ws)
        self._broadcast_socket = self._context.socket(zmq.PUB)
        self._request_socket = self._context.socket(zmq.REQ)
        self._response_socket = self._context.socket(zmq.REP)
        self._udp_socket = self._context.socket(zmq.RADIO)
        _, pending = await asyncio.wait(
            self.create_handler(ws),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await ws.close()
        await self.on_close(ws)

    async def _broadcast_handler(self):
        """
        The coroutine task for broadcasting messages on loop.
        """
        
        self._broadcast_socket.bind(f"tcp://{self.host}:{BROADCAST_PORT}")
        while self.on_start:
            await self._broadcast_socket.send_string(f"tcp://{self.host}:{BROADCAST_PORT}")
            await async_sleep(5)
        self._broadcast_socket.close()

    async def _request_handler(self):
        """
        The coroutine task for receiving request messages on loop.
        """
        self._request_socket = self._context.socket(zmq.REP)
        self._request_socket.bind(f"tcp://{self.host}:{REQ_PORT}")
        while self.on_start:
            msg = await self._request_socket.recv_string()
            await self._request_socket.send_string(msg)
        self._request_socket.close()

    def _server_task(self) -> None:
        """
        The thread task for running a service loop.
        """
        print(f"start a server on {self.host}:{self.port}")
        self._context = AsyncContext()
        asyncio.run(self.handler())
        self._context.term()
        print("server task finished")

    async def _send_msg_on_loop(
        self, 
        msg: MsgPack,
        ws: server.WebSocketServerProtocol, 
        sleep_time: float = 0
    ):
        """
        Send a str message
         
        It can only be used in the server thread.

        Args:
            msg (str): message to be sent.
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
            sleep_time (float, optional): sleep time after sending the message. Defaults to 0.
        """
        await ws.send(str.encode(str(msg)))
        await async_sleep(sleep_time)
 
    def send_msg(self, msg: MsgPack, sleep_time: float = 0) -> None:
        """
        Send a message outside the server thread.

        Args:
            msg (MsgPack): message to be sent.
            sleep_time (float, optional): sleep time after sending the message. Defaults to 0.
        """            
        if self._ws is None:
            return
        self.create_new_task(self._send_msg_on_loop(msg, self._ws, sleep_time))

    def create_new_task(self, task: asyncio.Task) -> None:
        """
        Create a new task on the service loop.

        Args:
            task (asyncio.Task): Task to be ruuning on the loop.
        """        
        loop = self._wsserver.get_loop()
        loop.create_task(task)

    def close_server(self) -> None:
        """
        Stop and close the server.
        """        
        self._wsserver.close()

    async def on_connect(self, ws: server.WebSocketServerProtocol) -> None:
        """
        This method will be executed once the connection is established.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """        
        print(f"connected by: {ws.local_address}")
        self._ws = ws

    async def on_close(self, ws: server.WebSocketServerProtocol) -> None:
        """
        This method will be executed when disconnected.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """        
        self._ws = None
        print(f"the connection to {ws.local_address} is closed")


class ObjPublisherBase(abc.ABC):
    """
    A abstract class for serializing simulation objects to json, and 
    then transmit their state by the server.
    
    Args:
        id (str): the id of this object.
    """    
    def __init__(
        self, 
        name: str,
    ) -> None:
        self.name: str = name

    @abc.abstractmethod
    def create_sim_obj(self) -> None:
        raise NotImplemented

    @abc.abstractmethod
    def update_obj_param(self) -> ObjData:
        raise NotImplemented

    @abc.abstractmethod
    def update_obj_state(self) -> ObjData:
        raise NotImplemented


if __name__ == "__main__":
    # s = SimPubData()
    pass