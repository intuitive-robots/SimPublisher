from flask import Blueprint, jsonify, request, render_template
from typing import Dict
from .utils import *
import traceback

main = Blueprint("main", __name__)

@main.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")


@main.route("/scan", methods=["GET"])
def scan():
    """Scan the network and store device data."""
    try:
        master_info, nodes_info = scan_network()
        master_info['type'] = 'master'  # Add type for master node
        scanned_data = {"status": "success", "master": master_info, "nodes": nodes_info}
        return jsonify(scanned_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to scan network: {str(e)}"}), 500


@main.route("/start-qr-alignment", methods=["POST"])
def start_qr_alignment():
    """Send QR calibration data to a specific node."""
    node_info: Dict = request.json  # type: ignore
    name = node_info.get("name")
    ip = node_info.get("ip")
    service_port = node_info.get("servicePort")
    if not ip or not service_port:
        return jsonify({"status": "error", "message": "IP and Service Port are required"}), 400
    # Read QR alignment data from the YAML file
    try:
        qr_data = read_qr_alignment_data("QRAlignment.yaml")
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "QRAlignment.yaml file not found"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading YAML: {str(e)}"}), 500
    # Send the QR data via ZeroMQ
    try:
        response = send_zmq_request(ip, service_port, f"{name}/StartQRAlignment", qr_data)
        return jsonify({"status": "success", "message": "QR Calibration successful", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during QR Calibration: {str(e)}"}), 500


@main.route("/stop-qr-alignment", methods=["POST"])
def stop_qr_alignment():
    """Stop the QR alignment process for a given IP and port."""
    node_info: Dict = request.json  # type: ignore
    name = node_info.get("name")
    ip = node_info.get("ip")
    service_port = node_info.get("servicePort")
    if not ip or not service_port:
        return jsonify({"status": "error", "message": "IP and Service Port are required"}), 400
    try:
        response = send_zmq_request(ip, int(service_port), f"{name}/StopQRAlignment", {})
        return jsonify({"status": "success", "message": "Stopped QR Alignment", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during Stop QR Alignment: {str(e)}"}), 500


@main.route('/rename-device', methods=['POST'])
def rename_device():
    data = request.get_json()
    new_name = data.get('newName')
    print(new_name)
    ip = data.get('ip')
    service_port = data.get('servicePort')
    if not ip or not service_port:
        return jsonify({"status": "error", "message": "IP and Service Port are required"}), 400
    try:
        response = send_zmq_request(ip, int(service_port), f"Rename", request=new_name)
        return jsonify({"status": "success", "message": "Rename Device", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during rename device: {str(e)}"}), 500


@main.route('/env-occlusion', methods=['POST'])
def env_occlusion():
    node_info: Dict = request.json  # type: ignore
    name = node_info.get("name")
    ip = node_info.get("ip")
    service_port = node_info.get("servicePort")
    if not ip or not service_port:
        return jsonify({"status": "error", "message": "IP and Service Port are required"}), 400
    try:
        response = send_zmq_request(ip, int(service_port), f"{name}/ToggleOcclusion", {})
        return jsonify({"status": "success", "message": "ToggleOcclusion", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during Toggle Occlusion: {str(e)}"}), 500
    
