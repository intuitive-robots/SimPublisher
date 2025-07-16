import zmq
import zmq.asyncio
import struct
import socket
from typing import List, TypedDict, Optional, Tuple
import enum
from traceback import print_exc

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


class NodeAddress(TypedDict):
    ip: IPAddress
    port: Port


def create_address(ip: IPAddress, port: Port) -> NodeAddress:
    return {"ip": ip, "port": port}


class MSG(enum.Enum):
    SERVICE_ERROR = b'\x10'
    SERVICE_TIMEOUT = b'\x11'

class EchoHeader:
    PING = b'\x00'
    HEARTBEAT = b'\x01'
    NODES = b'\x02'

class NodeInfo(TypedDict):
    name: str
    nodeID: str  # hash code since bytes is not JSON serializable
    addr: NodeAddress
    type: str
    servicePort: int
    topicPort: int
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


def get_zmq_socket_port(socket: zmq.asyncio.Socket) -> int:
    endpoint: bytes = socket.getsockopt(zmq.LAST_ENDPOINT)  # type: ignore
    return int(endpoint.decode().split(":")[-1])


def split_byte(bytes_msg: bytes) -> List[bytes]:
    return bytes_msg.split(b"|", 1)


def split_byte_to_str(bytes_msg: bytes) -> List[str]:
    return [item.decode() for item in split_byte(bytes_msg)]


def split_str(str_msg: str) -> List[str]:
    return str_msg.split("|", 1)


def search_for_master_node(
    local_ip: Optional[IPAddress] = None,
    search_time: int = 5,
    time_out: float = 0.1
) -> Optional[Tuple[IPAddress, str]]:
    if local_ip is not None:
        broadcast_ip = calculate_broadcast_addr(local_ip)
    else:
        broadcast_ip = "255.255.255.255"
        local_ip = "0.0.0.0"
    print(f"local_ip  {local_ip}")
    print(f"broadcast_ip {broadcast_ip}")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # 1. allow reuse so another process (e.g. the master) can also bind
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #    on recent Linux kernels you also need SO_REUSEPORT
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # 2. enable broadcast
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # 3. bind to all interfaces on the DISCOVERY_PORT
        s.bind(("", DISCOVERY_PORT))
        # 4. spray out a few PINGs
        for _ in range(search_time):
            s.sendto(EchoHeader.PING, ("192.168.178.255", DISCOVERY_PORT))
            s.settimeout(time_out)
            try:
                data, addr = s.recvfrom(1024)
                print(f"← got reply from {addr}: {data!r}")
                return addr[0], data.decode()
            except socket.timeout:
                continue

    print("No master node found.")
    return None
