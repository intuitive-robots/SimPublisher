
from .simpub_manager import ConnectionAbstract
from .log import logger


class Service(ConnectionAbstract):

    def __init__(self, service_name: str, _callback):
        super().__init__()
        self.service_name = service_name
        self._callback = _callback
        self.manager.service_callback[self.service_name] = _callback
        logger.info(f"Service {self.service_name} is ready")

    def on_shutdown(self):
        return super().on_shutdown()
