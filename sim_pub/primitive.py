import asyncio
from websockets import server
from time import localtime, strftime

from .model_loader.loader import XMLLoader
from .base import ServerBase, ObjPublisherBase

class PrimitiveServer(ServerBase):
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

class ObjectPublisher(ObjPublisherBase):
    """
    A new class for serializing simulation objects to json, and 
    then transmit their state by the server.

    Args:
        id (str): the id of this object.
        parent (ObjectSerializerBase, optional): the parent of this object. Defaults to None.
        child (ObjectSerializerBase, optional): the child of this object. Defaults to None.
    """    
    def __init__(
        self, 
        id: str,
        parent: ObjPublisherBase = None,
        child: ObjPublisherBase = None,
    ) -> None:
        super().__init__(id)
        self.parent = parent
        self.child = child


if __name__ == "__main__":
    # s = SimStreamer()
    # s.start_server_thread(block=True)
    pass