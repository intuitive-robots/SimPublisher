import zmq
import zmq.asyncio
import asyncio
import socket
from typing import List, TypedDict, Optional
import enum
from traceback import print_exc
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
MCAST_GRP = "239.192.1.1"


class NodeAddress(TypedDict):
    ip: IPAddress
    port: Port


def create_address(ip: IPAddress, port: Port) -> NodeAddress:
    return {"ip": ip, "port": port}


class MSG(enum.Enum):
    SERVICE_ERROR = b'\x10'
    SERVICE_TIMEOUT = b'\x11'


# class NodeInfo(TypedDict):
#     name: str
#     nodeID: str  # hash code since bytes is not JSON serializable
#     addr: NodeAddress
#     type: str
#     servicePort: int
#     topicPort: int
#     serviceList: List[ServiceName]
#     topicList: List[TopicName]

class ClientNodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    addr: NodeAddress
    type: str
    topicPort: int
    topicList: List[TopicName]


class XRNodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    ip: IPAddress
    type: str
    servicePort: int
    topicPort: int
    serviceList: List[ServiceName]
    topicList: List[TopicName]


def send_raw_request(messages: List[bytes], addr: str, timeout: int = 1000) -> Optional[bytes]:
    req_socket = zmq.Context().instance().socket(zmq.REQ)
    poller = zmq.Poller()
    poller.register(req_socket, zmq.POLLIN)
    req_socket.connect(addr)
    req_socket.setsockopt(zmq.RCVTIMEO, 200)
    start = time.time()
    try:
        req_socket.send_multipart(messages, copy=False)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        print_exc()
    start2 = time.time()
    result = None
    try:
        if poller.poll(timeout):
            result = req_socket.recv()
            print("Received:", result)
        else:
            print("Timeout occurred")
    except Exception as e:
        logger.error(
            f"Error when receiving message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        print_exc()
    finally:
        print(f"Message sent in {1000 * (start2 - start):.2f} ms, {1000 * (time.time() - start2):.2f} ms to receive response")
        req_socket.close()
        return result



def send_request(service_name: str, message: bytes, addr: str) -> Optional[bytes]:
    return send_raw_request([service_name.encode(), message], addr)


def send_string_request(service_name: str, message: str, addr: str) -> Optional[bytes]:
    return send_raw_request([service_name.encode(), message.encode()], addr)


async def send_raw_request_async(messages: List[bytes], addr: str) -> bytes:
    req_socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
    req_socket.connect(addr)
    try:
        start = time.time()
        await req_socket.send_multipart(messages, copy=False)
        start2 = time.time()
        print(f"message: {messages[0].decode()}, addr: {addr}")
        # Set receive timeout to 200ms (0.2 seconds)
        result = await asyncio.wait_for(req_socket.recv(), timeout=0.2)
        print(f"Message sent in {1000 * (start2 - start):.2f} ms, {1000 * (time.time() - start2):.2f} ms to receive response")
        req_socket.close()
        return result
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        req_socket.close()

# def calculate_broadcast_addr(ip_addr: IPAddress) -> IPAddress:
#     ip_bin = struct.unpack("!I", socket.inet_aton(ip_addr))[0]
#     netmask_bin = struct.unpack("!I", socket.inet_aton("255.255.255.0"))[0]
#     broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
#     return socket.inet_ntoa(struct.pack("!I", broadcast_bin))


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


# def search_for_master_node(
#     local_ip: Optional[IPAddress] = None,
#     search_time: int = 5,
#     time_out: float = 0.1
# ) -> Optional[Tuple[IPAddress, str]]:
#     if local_ip is not None:
#         broadcast_ip = calculate_broadcast_addr(local_ip)
#     else:
#         broadcast_ip = "255.255.255.255"
#     with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _socket:
#         # wait for response
#         _socket.bind(("0.0.0.0", 0))
#         _socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
#         for _ in range(search_time):
#             _socket.sendto(
#                 EchoHeader.PING.value, (broadcast_ip, DISCOVERY_PORT)
#             )
#             _socket.settimeout(time_out)
#             try:
#                 data, addr = _socket.recvfrom(1024)
#                 logger.info(f"Find a master node at {addr[0]}:{addr[1]}")
#                 return addr[0], data.decode()
#             except socket.timeout:
#                 continue
#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 logger.error(f"Error when searching for master node: {e}")
#                 print_exc()
#     logger.info("No master node found, start as master node")
#     return None
