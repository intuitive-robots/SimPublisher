from .log import logger

from .net_manager import AsyncSocket


async def send_message(msg: str, socket: AsyncSocket):
    try:
        await socket.send_string(msg)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
    return await socket.recv_string()
