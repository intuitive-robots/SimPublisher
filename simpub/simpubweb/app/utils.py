from __future__ import annotations

from typing import Dict

import yaml
import zmq
from json import dumps


def read_qr_alignment_data(filepath: str) -> dict:
    """Read QR alignment data from a YAML file."""
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def send_zmq_request(
    ip: str, port: int, service_name: str, request: Dict, timeout: int = 1000
) -> str:
    """Send a ZMQ REQ/REP request to the given IP and port."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    try:
        socket.connect(f"tcp://{ip}:{port}")
        socket.setsockopt(zmq.RCVTIMEO, timeout)
        socket.setsockopt(zmq.SNDTIMEO, timeout)
        if isinstance(request, str):
            socket.send_string("".join([service_name, "|", request]))
        else:
            socket.send_string("".join([service_name, "|", dumps(request)]))
        response = socket.recv_string()
        if response == "NOSERVICE":
            raise Exception("Service not found on the node.")
        return response
    except zmq.ZMQError as exc:
        raise zmq.ZMQError from exc
    finally:
        socket.close()
