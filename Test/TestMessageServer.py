from simpub.core.simpub_server import MsgServer
from simpub.core.net_manager import Service 
from simpub.core.log import logger

class TestMsgServer(MsgServer):

    def __init__(self, host: str = "127.0.0.1"):
        super().__init__(host)
        self.save_record_service = Service("SaveRecord", self._on_save_record, str)
        
    def _on_save_record(self, record: str) -> str:
        logger.info(f"Received record: {record}")
        return "Successfully saved record"

if __name__ == "__main__":
    server = TestMsgServer(host="127.0.0.1")
    server.join()
