from websockets import server
import threading
import abc
import asyncio

class ServerBase(abc.ABC):
    """
    A abstract class for all server running on a simulation environment.

    The server receives and sends messages to client by websockets.
    """    
    def __init__(self, host: str) -> None:
        self._server_thread: threading.Thread = None
        self._loop: asyncio.AbstractEventLoop = None

        self._host = host
        self._is_active: bool = False

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

    @abc.abstractmethod
    def _server_task(self) -> None:
        """
        The thread task for running a service loop.
        """        
        raise NotImplementedError

    @abc.abstractmethod
    def send_request(self, msg: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def close_server(self) -> None:
        """
        Stop and close the server.
        """        
        raise NotImplementedError
