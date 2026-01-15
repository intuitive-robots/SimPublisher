from __future__ import annotations

from json import dumps
import json
from typing import Dict

from pathlib import Path
import yaml
import zmq


def read_qr_alignment_data(filepath: str) -> dict:
    """Read QR alignment data from a YAML file."""
    
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def create_scene_config_file(scene_data: dict = None, output_filepath: Path = Path("Scene.json")) -> None:
    """Create a scene configuration YAML file.
    Args:
        scene_data (dict, optional): Scene configuration data. If None, a default scene is created.
        output_filepath (str): Path to the output YAML file. 
    """
    if scene_data is None:
        scene_data = [
            {
                "name": "MujocoScene",
                "qrCode": "IRIS",
                "offset": { "x": 0, "y": 0, "z": -1, "rotX": 0, "rotY": 0, "rotZ": 0 }
            }
        ]
    with open(output_filepath, "w") as file:
        json.dump(scene_data, file)

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
