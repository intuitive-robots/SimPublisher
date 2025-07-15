import zmq
import zmq.asyncio
import asyncio
import socket
from typing import List, TypedDict, Optional, Callable, Any
import enum
from traceback import print_exc
from functools import wraps
import time

from .log import logger

IPAddress = str
Port = int
TopicName = str
ServiceName = str
AsyncSocket = zmq.asyncio.Socket
HashIdentifier = str

BROADCAST_INTERVAL = 0.5
HEARTBEAT_INTERVAL = 0.2
DISCOVERY_PORT = int(7720)
MULTICAST_GRP = "239.255.10.10"


class MSG(enum.Enum):
    SERVICE_ERROR = b"\x10"
    SERVICE_TIMEOUT = b"\x11"


class XRNodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    ip: IPAddress
    type: str
    servicePort: int
    topicPort: int
    serviceList: List[ServiceName]
    topicList: List[TopicName]


def request_log(func: Callable) -> Callable:
    """
    Decorator that logs execution time and message sizes in KB
    Shows the request name from the first message instead of function name
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()

        # Calculate input message size and extract request name
        input_size_kb = 0
        request_name = "unknown"
        if args and isinstance(args[0], list):
            # First argument should be messages: List[bytes]
            messages = args[0]
            total_bytes = sum(
                len(msg) for msg in messages if isinstance(msg, bytes)
            )
            input_size_kb = total_bytes / 1024
            # Extract request name from first message
            if messages and isinstance(messages[0], bytes):
                try:
                    request_name = messages[0].decode("utf-8")
                except UnicodeDecodeError:
                    # If decode fails, show as binary with length
                    request_name = f"<binary:{len(messages[0])}bytes>"

        try:
            result = await func(*args, **kwargs)
            total_time = time.time() - start_time

            # Calculate output message size
            output_size_kb = 0
            if result and isinstance(result, bytes):
                output_size_kb = len(result) / 1024

            logger.info(
                f"Request '{request_name}' took {total_time*1000:.2f}ms, "
                f"sent: {input_size_kb:.2f}KB, "
                f"received: {output_size_kb:.2f}KB"
            )
            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                f"Request '{request_name}' failed after "
                f"{total_time*1000:.2f}ms, "
                f"sent: {input_size_kb:.2f}KB: {e}"
            )
            raise

    return wrapper


@request_log
async def send_raw_request_async(
    messages: List[bytes], addr: str, timeout: int = 2
) -> Optional[bytes]:
    req_socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
    req_socket.connect(addr)
    result = None
    try:
        await req_socket.send_multipart(messages, copy=False)
        result = await asyncio.wait_for(req_socket.recv(), timeout=timeout)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        print_exc()
    finally:
        req_socket.close()
        return result


async def send_string_request_async(
    messages: List[str], addr: str, timeout: int = 2
) -> Optional[bytes]:
    return await send_raw_request_async(
        [item.encode() for item in messages], addr, timeout
    )


def create_udp_socket() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def get_zmq_socket_port(socket: zmq.asyncio.Socket) -> int:
    endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
    return int(endpoint.decode().split(":")[-1])


def get_zmq_socket_url(socket: zmq.asyncio.Socket) -> str:
    endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
    return endpoint.decode()


def split_byte(bytes_msg: bytes) -> List[bytes]:
    return bytes_msg.split(b"|", 1)


def split_byte_to_str(bytes_msg: bytes) -> List[str]:
    return [item.decode() for item in split_byte(bytes_msg)]


def split_str(str_msg: str) -> List[str]:
    return str_msg.split("|", 1)
