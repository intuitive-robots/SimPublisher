from websockets import server
import asyncio
from asyncio import sleep as async_sleep
import threading
import json
import abc
from typing import TypedDict, Dict, List


class SimPubDataBlock(TypedDict):
    str_dict: Dict[str, str]
    list_dict: Dict[str, List[float]]
    bool_dict: Dict[str, bool]

SimPubData = Dict[str, SimPubDataBlock]


class SimPubMsg(TypedDict):
    header: str
    data: SimPubData


class ServerBase(abc.ABC):
    """
    A abstract class for all server running on a simulation environment.

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

        self._ws: server.WebSocketServerProtocol = None
        self._wsserver: server.WebSocketServer = None
        self._server_future: asyncio.Future = None
        self.host = host
        self.port = port
        
    def start_server_thread(self, block = False) -> None:
        """
        Start a thread for service.

        Args:
            block (bool, optional): main thread stop running and 
            wait for server thread. Defaults to False.
        """
        self.server_thread = threading.Thread(target=self._start_server_task)
        self.server_thread.daemon = True
        self.server_thread.start()
        if block:
            self.server_thread.join()

    @abc.abstractmethod
    async def process_message(self, msg:str):
        """
        Processing message when receives new messages from clients.

        Args:
            msg (str): message from client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_handler(self, ws: server.WebSocketServerProtocol) -> list[asyncio.Task]:
        """
        Create new tasks such as receiving and stream when start service.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.

        Returns:
            list[asyncio.Task]: the list of new tasks
        """        
        raise NotImplementedError

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
        await self.on_conncet(ws)
        _, pending = await asyncio.wait(
            self.create_handler(ws),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await ws.close()
        await self.on_close(ws)

    async def _expect_client(self):
        """
        The coroutine task for running service on loop.
        """        
        self.stop = asyncio.Future()
        async with server.serve(self.handler, self.host, self.port) as self._wsserver:
            await self.stop

    def _start_server_task(self) -> None:
        """
        The thread task for running a service loop.
        """        
        print(f"start a server on {self.host}:{self.port}")
        asyncio.run(self._expect_client())
        print("server task finished")

    async def _send_str_msg_on_loop(
        self, msg: str, 
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
        await ws.send(str.encode(json.dumps(msg)))
        await async_sleep(sleep_time)
 
    def send_str_msg(self, msg: str, sleep_time: float = 0) -> None:
        """
        Send a str message outside the server thread.

        Args:
            msg (str): message to be sent.
            sleep_time (float, optional): sleep time after sending the message. Defaults to 0.
        """            
        if self._ws is None:
            return
        self.create_new_task(self._send_str_msg_on_loop(msg, self._ws, sleep_time))

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

    async def on_conncet(self, ws: server.WebSocketServerProtocol) -> None:
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

class ObjectPublisherBase(abc.ABC):
    """
    A abstract class for serializing simulation objects to json, and 
    then transmit their state by the server.
    
    Args:
        id (str): the id of this object.
    """    
    def __init__(
        self, 
        id: str,
    ) -> None:
        self.id: str = id

    @abc.abstractmethod
    def get_obj_param_dict(self) -> dict:
        raise NotImplemented

    @abc.abstractmethod
    def get_obj_state_dict(self) -> dict:
        raise NotImplemented

