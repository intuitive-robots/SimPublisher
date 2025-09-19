import zmq
import zmq.asyncio
import asyncio
import socket
from typing import List, TypedDict, Optional, Callable, Any, Dict, Tuple
import enum
from traceback import print_exc
from functools import wraps
import time
from dataclasses import dataclass, field

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


class IRISMSG(enum.Enum):
    EMPTY = "EMPTY"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    NOTFOUND = "NOTFOUND"
    START = "START"
    STOP = "STOP"


class XRNodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    nodeInfoID: str  # hash code since bytes is not JSON serializable
    ip: IPAddress
    type: str
    port: int
    serviceList: List[ServiceName]
    topicDict: Dict[TopicName, int]


def _format_services(services):
    if not services:
        return "    (none)"
    return "\n".join(f"    • {s}" for s in sorted(services))


def _format_topics(topics):
    if not topics:
        return "    (none)"
    # sort by topic name
    return "\n".join(
        f"    • {k}: {v}"
        for k, v in sorted(topics.items(), key=lambda kv: kv[0])
    )


def _box(text: str, title: str = "") -> str:
    # simple unicode box with optional title
    lines = text.splitlines()
    inner_width = max(len(line) for line in lines) if lines else 0
    if title:
        header = (
            f"┌─ {title} " + "─" * max(0, inner_width - len(title) - 1) + "┐"
        )
    else:
        header = "┌" + "─" * inner_width + "┐"
    body = "\n".join("│" + line.ljust(inner_width) + "│" for line in lines)
    footer = "└" + "─" * inner_width + "┘"
    return f"{header}\n{body}\n{footer}"


def print_node_info(node_info: XRNodeInfo):
    name = node_info.get("name", "?")
    nodeID = node_info.get("nodeID", "?")
    ip = node_info.get("ip", "?")
    port = node_info.get("port", "?")
    svcs = node_info.get("serviceList", []) or node_info.get("services", [])
    topics = node_info.get("topicDict", {}) or node_info.get("topics", {})

    content = (
        f"Node     : {name}\n"
        f"Node ID  : {nodeID}\n"
        f"Address  : {ip}:{port}\n"
        f"Services : {len(svcs)}\n"
        f"{_format_services(svcs)}\n"
        f"Topics   : {len(topics)}\n"
        f"{_format_topics(topics)}"
    )
    logger.info("\n" + _box(content, title="Node Info"))


@dataclass
class XRNodeEntry:
    """Store XR node metadata alongside its last heartbeat."""

    info: Optional[XRNodeInfo] = None
    last_heartbeat: float = field(default_factory=lambda: 0.0)

    def touch(self) -> None:
        self.last_heartbeat = time.time()

    def update_info(self, info: XRNodeInfo) -> None:
        self.info = info
        self.touch()


class XRNodeRegistry:
    """Registry holding node information and heartbeat timestamps."""

    def __init__(self) -> None:
        self._records: Dict[str, XRNodeEntry] = {}

    def get(self, node_id: str) -> Optional[XRNodeEntry]:
        return self._records.get(node_id)

    def touch(self, node_id: str) -> XRNodeEntry:
        record = self._records.setdefault(node_id, XRNodeEntry())
        record.touch()
        return record

    def update_info(self, node_id: str, info: XRNodeInfo) -> XRNodeEntry:
        record = self._records.setdefault(node_id, XRNodeEntry())
        record.update_info(info)
        return record

    def remove(self, node_id: str) -> None:
        self._records.pop(node_id, None)

    def remove_offline(self, timeout: float) -> List[Tuple[str, XRNodeEntry]]:
        now = time.time()
        removed: List[Tuple[str, XRNodeEntry]] = []
        for node_id, record in list(self._records.items()):
            if record.last_heartbeat and now - record.last_heartbeat > timeout:
                removed.append((node_id, record))
                self._records.pop(node_id, None)
        return removed

    def items(self):
        return self._records.items()

    def values(self):
        return self._records.values()

    def registered_infos(self) -> List[XRNodeInfo]:
        return [
            record.info for record in self._records.values() if record.info
        ]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._records


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
                f"speed: {input_size_kb / 1024 / total_time:.2f}MB/s, "
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
async def send_request_async(
    messages: List[str], req_socket: AsyncSocket, timeout: int = 2
) -> Optional[bytes]:
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


async def send_request_with_addr_async(
    messages: List[bytes], addr: str, timeout: int = 2
) -> Optional[bytes]:
    req_socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
    req_socket.connect(addr)
    return await send_request_async(messages, req_socket, timeout)


async def send_string_request_async(
    messages: List[str], addr: str, timeout: int = 2
) -> Optional[bytes]:
    return await send_request_with_addr_async(
        [item.encode() for item in messages], addr, timeout
    )


def send_raw_request_with_addr(
    messages: List[bytes], addr: str, timeout: int = 2
) -> Optional[str]:
    """Send a REQ to an address with timeout to avoid blocking forever."""
    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(addr)

    # Set receive timeout (milliseconds)
    req_socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)

    result = None
    try:
        req_socket.send_multipart(messages, copy=False)
        result = req_socket.recv().decode()
    except zmq.error.Again:
        logger.error("Timeout reached while waiting for reply")
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        print_exc()
    finally:
        req_socket.close()
        return result


def send_request_with_addr(
    request_name: str, request: str, addr: str, timeout: int = 2
) -> Optional[str]:
    return send_raw_request_with_addr(
        [request_name.encode(), request.encode()], addr, timeout
    )


def create_udp_socket() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def get_zmq_socket_url(socket: zmq.asyncio.Socket) -> str:
    endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
    return endpoint.decode()


# def get_zmq_socket_port(socket: zmq.asyncio.Socket) -> int:
#     endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
#     return int(endpoint.decode().split(":")[-1])


# def split_byte(bytes_msg: bytes) -> List[bytes]:
#     return bytes_msg.split(b"|", 1)


# def split_byte_to_str(bytes_msg: bytes) -> List[str]:
#     return [item.decode() for item in split_byte(bytes_msg)]


# def split_str(str_msg: str) -> List[str]:
#     return str_msg.split("|", 1)
