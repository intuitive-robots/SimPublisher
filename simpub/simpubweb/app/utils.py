from typing import List, Dict
import enum
import yaml
import zmq
from json import dumps, loads
import socket


class EchoHeader(enum.Enum):
    PING = b'\x00'
    HEARTBEAT = b'\x01'
    NODES = b'\x02'


def split_byte(bytes_msg: bytes) -> List[bytes]:
    return bytes_msg.split(b"|", 1)


def split_byte_to_str(bytes_msg: bytes) -> List[str]:
    return [item.decode() for item in split_byte(bytes_msg)]


def split_str(str_msg: str) -> List[str]:
    return str_msg.split("|", 1)


def read_qr_alignment_data(filepath: str) -> dict:
    """Read QR alignment data from a YAML file."""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def scan_network() -> tuple[Dict, Dict]:
    
    _socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _socket.bind(("0.0.0.0", 7720))
    _socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _socket.settimeout(5)
    print("Listening on 0.0.0.0:7720 for broadcast messages...")
    for _ in range(5):
        try:
            data, _ = _socket.recvfrom(4096)
            msg = data.decode()
            if msg.startswith("SimPub"):
                break
        except socket.timeout:
            print(f"No messages received within 5 seconds. Continuing to listen...")
        except KeyboardInterrupt:
            print("Stopping listener...")
        except Exception as e:
            print(f"Error: {e}")
    _socket.close()
    master_info = loads(msg.split("|", maxsplit=2)[1])
    ip_addr = master_info["addr"]["ip"]
    service_port = master_info["servicePort"]
    msg = send_zmq_request(ip_addr, service_port, "GetNodesInfo", {})
    nodes_info = loads(msg)
    return master_info, nodes_info


def send_zmq_request(
    ip: str, port: int, service_name: str, request: Dict, timeout: int = 1000
) -> str:
    """
    Send a ZMQ request to the given IP and port with a specified timeout.
    
    :param ip: Target IP address.
    :param port: Target port number.
    :param service_name: The name of the service to identify the request.
    :param request: The request payload as a dictionary.
    :param timeout: Timeout in milliseconds for send and receive operations (default: 5000ms).
    :return: Response string from the server.
    :raises zmq.ZMQError: If a timeout occurs or there is a ZMQ-related error.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    try:
        socket.connect(f"tcp://{ip}:{port}")
        
        # Set timeout options
        socket.setsockopt(zmq.RCVTIMEO, timeout)  # Receive timeout
        socket.setsockopt(zmq.SNDTIMEO, timeout)  # Send timeout
        
        # Send the request
        if type(request) == str:
            socket.send_string("".join([service_name, "|", request]))
        else:
            socket.send_string("".join([service_name, "|", dumps(request)]))
        
        
        # Receive the response
        response = socket.recv_string()
        if response == 'NOSERVICE':  # type: ignore
            raise Exception("Service not found on the node.")
        return response  # type: ignore
    except zmq.ZMQError as e:
        raise zmq.ZMQError
    except Exception as e:
        raise e
    finally:
        socket.close()