import zmq
import zmq.asyncio
import struct
import socket
from typing import List, Tuple, TypedDict, Optional
from json import dumps
import enum

from .log import logger

IPAddress = str
TopicName = str
ServiceName = str
AsyncSocket = zmq.asyncio.Socket
HashIdentifier = str

BROADCAST_INTERVAL = 0.5
HEARTBEAT_INTERVAL = 0.2
DISCOVERY_PORT = 7720


class MSG(enum.Enum):
    PING = b'\x01'
    PING_ACK = b'\x02'
    SERVICE_ERROR = b'\x03'
    SERVICE_TIMEOUT = b'\x04'


class NodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    ip: str
    type: str
    servicePort: str
    topicPort: str
    serviceList: List[ServiceName]
    topicList: List[TopicName]


async def send_request(
    msg: str,
    addr: str,
    context: zmq.asyncio.Context
) -> str:
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(addr)
    try:
        await req_socket.send_string(msg)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
    result = await req_socket.recv_string()
    req_socket.close()
    return result


def calculate_broadcast_addr(ip_addr: IPAddress) -> IPAddress:
    ip_bin = struct.unpack("!I", socket.inet_aton(ip_addr))[0]
    netmask_bin = struct.unpack("!I", socket.inet_aton("255.255.255.0"))[0]
    broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
    return socket.inet_ntoa(struct.pack("!I", broadcast_bin))


def create_udp_socket() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def get_socket_port(socket: zmq.asyncio.Socket) -> str:
    endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
    return endpoint.decode().split(":")[-1]


def split_byte(bytes_msg: bytes) -> List[bytes]:
    return bytes_msg.split(b":", 1)


def split_byte_to_str(bytes_msg: bytes) -> List[str]:
    return [item.decode() for item in split_byte(bytes_msg)]


def generate_node_msg(node_info: NodeInfo) -> bytes:
    return b"".join(
        [node_info["nodeID"].encode(), b":", dumps(node_info).encode()]
    )


def search_for_master_node(
    local_ip: Optional[IPAddress] = None
) -> Optional[Tuple[IPAddress, str]]:
    if local_ip is not None:
        broadcast_ip = calculate_broadcast_addr(local_ip)
    else:
        broadcast_ip = "255.255.255.255"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _socket:
        # wait for response
        _socket.bind(("0.0.0.0", 0))
        for _ in range(5):
            # _, local_port = _socket.getsockname()
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            _socket.sendto(MSG.PING.value, (broadcast_ip, DISCOVERY_PORT))
            _socket.settimeout(0.1)
            try:
                data, addr = _socket.recvfrom(1024)
                logger.info(f"Find a master node at {addr[0]}:{addr[1]}")
                return addr[0], data.decode()
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    logger.info("No master node found, start as master node")
    return None


def print_node_info(node_info: NodeInfo):
    print(f"Node name: {node_info['name']}")
    print(f"    Node ID: {node_info['nodeID']}")
    print(f"    Node IP: {node_info['ip']}")
    print(f"    Node type: {node_info['type']}")
    print(f"    Node service port: {node_info['servicePort']}")
    print(f"    Node topic port: {node_info['topicPort']}")
    print(f"    Node service list: {node_info['serviceList']}")
    print(f"    Node topic list: {node_info['topicList']}")