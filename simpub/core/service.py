from typing import Callable
import zmq

from .net_manager import ConnectionAbstract
from .log import logger


class Service(ConnectionAbstract):

    def __init__(
        self,
        service_name: str,
        callback: Callable[[str, zmq.Socket], None]
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.manager.register_local_service(
            service_name, callback
        )
        logger.info(f"Service {self.service_name} is ready")

    def on_shutdown(self):
        return super().on_shutdown()
