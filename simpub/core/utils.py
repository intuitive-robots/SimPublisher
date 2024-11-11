import zmq
from .log import logger


async def send_message(msg: str, socket: zmq.Socket):
    try:
        await socket.send_string(msg)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
    return await socket.recv_string()
