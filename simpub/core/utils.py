import zmq
import zmq.asyncio
import socket
from typing import List, Dict, Tuple, TypedDict, Optional
import enum
from traceback import print_exc
from json import loads, dumps
import psutil

from .log import logger

IPAddress = str
Port = int
TopicName = str
ServiceName = str
AsyncSocket = zmq.asyncio.Socket
HashIdentifier = str

BROADCAST_INTERVAL = 0.5
HEARTBEAT_INTERVAL = 0.2
INTERNAL_DISCOVER_PORT = int(7720)
EXTERNAL_BROADCAST_PORT = int(7721)
EXTERNAL_DISCOVER_PORT = int(7723)

class MSG(enum.Enum):
    SERVICE_SUCCESS = b"\x00"
    SERVICE_ERROR = b"\x10"
    SERVICE_TIMEOUT = b"\x11"

class NodeAddress(TypedDict):
    ip: IPAddress
    port: Port

class NodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    addr: NodeAddress
    type: str
    servicePort: int
    topicPort: int
    services: List[ServiceName]
    topics: List[TopicName]



def create_address(ip: IPAddress, port: Port) -> NodeAddress:
    return {"ip": ip, "port": port}




async def send_request(msg: str, addr: str, context: zmq.asyncio.Context) -> str:
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(addr)
    try:
        await req_socket.send_string(msg)
    except Exception as e:
        logger.error(
            f"Error when sending message from send_message function in "
            f"simpub.core.utils: {e}"
        )
        print_exc()
    result = await req_socket.recv_string()
    req_socket.close()
    return result


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


def split_byte(bytes_msg: bytes) -> List[bytes]:
    return bytes_msg.split(b"|", 1)


def split_byte_to_str(bytes_msg: bytes) -> List[str]:
    return [item.decode() for item in split_byte(bytes_msg)]


def split_str(str_msg: str) -> List[str]:
    return str_msg.split("|", 1)


def is_udp_port_in_use(port, host="127.0.0.1"):
    with create_udp_socket() as s:
        try:
            s.bind((host, port))
        except socket.error:
            return True
        return False


def is_local_port_in_use(ip: str, port: int):
    """
    Check if a specific port is in use on the local system using psutil.

    Args:
        port (int): The port number to check.

    Returns:
        bool: True if the port is in use, False otherwise.
    """
    # Get a list of all connections
    connections = psutil.net_connections(kind="inet")
    # Check if the port is in use
    for conn in connections:
        if conn.laddr and conn.laddr.ip == ip and conn.laddr.port == port:
            return True
    return False


# def search_for_master_node_external(
#     local_ip: Optional[IPAddress] = None, search_time: int = 5, time_out: float = 0.1
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
#             _socket.sendto(EchoHeader.PING.value, (broadcast_ip, DISCOVERY_PORT))
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


def search_for_master_node_internal(host_ip: str) -> Optional[NetInfo]:
    if not is_local_port_in_use(host_ip, INTERNAL_DISCOVER_PORT):
        return None
    sub_socket = zmq.Context().socket(zmq.SUB)
    sub_socket.connect(f"tcp://{host_ip}:{INTERNAL_DISCOVER_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    msg = sub_socket.recv_string()
    if not msg.startswith("SimPub"):
        raise ValueError("Received message is not a SimPub message.")
    sub_socket.close()
    return loads(msg[6:])


# def search_for_master_node_external() -> tuple[Dict, Dict]:
#     _socket = create_udp_socket()
#     _socket.bind(("0.0.0.0", 0))
#     _socket.settimeout(5)
#     print("Listening on 0.0.0.0:7720 for broadcast messages...")
#     for _ in range(5):
#         try:
#             data, _ = _socket.recvfrom(4096)
#             msg = data.decode()
#             if msg.startswith("SimPub"):
#                 break
#         except socket.timeout:
#             print(f"No messages received within 5 seconds. Continuing to listen...")
#         except KeyboardInterrupt:
#             print("Stopping listener...")
#         except Exception as e:
#             print(f"Error: {e}")
#     _socket.close()
#     master_info = loads(msg.split("|", maxsplit=2)[1])
#     ip_addr = master_info["addr"]["ip"]
#     service_port = master_info["servicePort"]
#     msg = send_zmq_request(ip_addr, service_port, "GetNodesInfo", {})
#     nodes_info = loads(msg)
#     return master_info, nodes_info
