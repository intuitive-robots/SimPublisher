import asyncio
from websockets import server

from ...model_loader.loader import XMLLoader
from ..base import ServerBase

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



    async def receive_handler(self, ws: server.WebSocketServerProtocol):
        """
        Default messages receiving task handler.

        Args:
        ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """        
        async for msg in ws:
            await self.process_message(msg)

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

    async def _send_msg_on_loop(
        self, msg: MsgPack,
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
 
    def send_request(self, msg: MsgPack, sleep_time: float = 0) -> None:
        """
        Send a message outside the server thread.

        Args:
            msg (MsgPack): message to be sent.
            sleep_time (float, optional): sleep time after sending the message. Defaults to 0.
        """            
        if self._ws is None:
            return
        self.create_new_task(self._send_msg_on_loop(msg, self._ws, sleep_time))




class WsServer(ServerBase):
    """
    A basic implementation of server for receiving and printing messages.
    """    
    def __init__(
        self,
        host="127.0.0.1",
        port=8052,
    ) -> None:
        super().__init__(host, port)

    async def process_message(self, msg:str):
        print(f"new message: {msg}")

    def create_handler(self, ws: server.WebSocketServerProtocol) -> list[asyncio.Task]:
        """
        Create a new receiving task for server.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.

        Returns:
            list[asyncio.Task]: the list of new tasks
        """        
        return [asyncio.create_task(self.receive_handler(ws))]

    async def handler(self, ws: server.WebSocketServerProtocol):
        """
        The main task handler of loop in server thread.

        Args:
            ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """
        await self.on_connect(ws)
        _, pending = await asyncio.wait(
            self.create_handler(ws),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await ws.close()
        await self.on_close(ws)



class SimStreamer(PrimitiveServer):
    """
    A basic server for streaming data and receiving messages.

    Args:
        dt (float): message stream interval (second). Defaults to 0.05.
        on_stream (bool): whether starting stream right now.
    """    
    def __init__(
            self,
            dt: float =  0.05,
            publisher_list: list[ObjPublisherBase] = list(),
            host = "127.0.0.1", 
            port = 8052,
            on_stream: bool = False,
        ) -> None:
        super().__init__(host, port)
        self.publisher_list = publisher_list
        # flags
        self.on_stream = on_stream

        # stream frequency
        self.dt = dt

    def update_stream_data(self) -> str:
        """
        Update data for stream.

        Returns:
            str: the data to be sent
        """
        return [item.update_obj_state() for item in self.publisher_list]

    def start_stream(self) -> None:
        """
        Start stream.
        """        
        if self._ws is None:
            return
        self.create_new_task(self.stream_handler(self._ws, self.dt))
        self.on_stream = True

    def close_stream(self) -> None:
        """
        Close stream.
        """        
        self.on_stream = False

    async def on_start_stream(self):
        """
        This method will be executed once the stream is started.
        """        
        print("start stream")

    async def on_close_stream(self):
        """
        This method will be executed once the stream is closed
        """        
        self.on_stream = False
        print("close stream")

    async def stream_handler(self, ws: server.WebSocketServerProtocol, dt:float):
        """
        Default message stream task handler.

        Args:
        ws (server.WebSocketServerProtocol): WebSocketServerProtocol from websockets.
        """
        await self.on_start_stream()
        while self.on_stream:
            stream_data = self.update_stream_data()
            try:
                pass
                # await self._send_str_msg_on_loop(stream_data, ws, dt)
            except:
                print("error occured when sending messages!!!!!")
                await ws.close()
                break
            finally:
                pass
        await self.on_close_stream()
