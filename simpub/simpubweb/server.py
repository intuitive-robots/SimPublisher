from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.serving import BaseWSGIServer, make_server

from ..core.utils import XRNodeRegistry
from .app.utils import read_qr_alignment_data, send_zmq_request


class SimPubWebServer:
    """Flask-based web interface backed by an XR node registry."""

    def __init__(
        self,
        xr_nodes: XRNodeRegistry,
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        self.xr_nodes = xr_nodes
        self.host = host
        self.port = port
        template_dir = Path(__file__).resolve().parent / "app" / "templates"
        self.app = Flask(__name__, template_folder=str(template_dir))
        self.app.config["XR_NODE_REGISTRY"] = xr_nodes
        self._server: Optional[BaseWSGIServer] = None
        self._context = None
        self._started: bool = False
        self._register_routes()

    # ------------------------------------------------------------------
    # Route registration and handlers
    # ------------------------------------------------------------------
    def _register_routes(self) -> None:
        self.app.route("/", methods=["GET"])(self.index)
        self.app.route("/scan", methods=["GET"])(self.scan)
        self.app.route("/teleport-scene", methods=["POST"])(
            self.start_qr_alignment
        )
        self.app.route("/stop-qr-alignment", methods=["POST"])(
            self.stop_qr_alignment
        )
        self.app.route("/rename-device", methods=["POST"])(self.rename_device)
        self.app.route("/env-occlusion", methods=["POST"])(self.env_occlusion)

    def index(self):
        return render_template("index.html")

    def scan(self):
        try:
            nodes = []
            for node in self.xr_nodes.registered_infos():
                node_payload = dict(node)
                node_payload.setdefault("ip", node.get("ip"))
                node_payload["servicePort"] = node_payload.get("port")
                nodes.append(node_payload)
            return jsonify({"status": "success", "nodes": nodes})
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to fetch nodes: {exc}",
                    }
                ),
                500,
            )

    def start_qr_alignment(self):
        payload = request.get_json(silent=True) or {}
        try:
            name, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            qr_data = read_qr_alignment_data("QRAlignment.yaml")
        except FileNotFoundError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "QRAlignment.yaml file not found",
                    }
                ),
                500,
            )
        except Exception as exc:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error reading YAML: {exc}",
                    }
                ),
                500,
            )
        try:
            response = send_zmq_request(
                ip, service_port, f"{name}/StartQRAlignment", qr_data
            )
            return jsonify(
                {
                    "status": "success",
                    "message": "QR Calibration successful",
                    "response": response,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error during QR Calibration: {exc}",
                    }
                ),
                500,
            )

    def stop_qr_alignment(self):
        payload = request.get_json(silent=True) or {}
        try:
            name, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            response = send_zmq_request(
                ip, service_port, f"{name}/StopQRAlignment", {}
            )
            return jsonify(
                {
                    "status": "success",
                    "message": "Stopped QR Alignment",
                    "response": response,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error during Stop QR Alignment: {exc}",
                    }
                ),
                500,
            )

    def rename_device(self):
        payload = request.get_json(silent=True) or {}
        new_name = payload.get("newName")
        if not new_name:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "New name is required",
                    }
                ),
                400,
            )
        try:
            _, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            response = send_zmq_request(
                ip, service_port, "Rename", request=new_name
            )
            return jsonify(
                {
                    "status": "success",
                    "message": "Rename Device",
                    "response": response,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error during rename device: {exc}",
                    }
                ),
                500,
            )

    def env_occlusion(self):
        payload = request.get_json(silent=True) or {}
        try:
            name, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            response = send_zmq_request(
                ip, service_port, f"{name}/ToggleOcclusion", {}
            )
            return jsonify(
                {
                    "status": "success",
                    "message": "ToggleOcclusion",
                    "response": response,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error during Toggle Occlusion: {exc}",
                    }
                ),
                500,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_connection_info(
        self, payload: dict[str, object]
    ) -> tuple[str, str, int]:
        name = payload.get("name")
        ip = payload.get("ip")
        service_port = payload.get("servicePort", payload.get("port"))
        if not name:
            raise ValueError("Device name is required")
        if not ip:
            raise ValueError("IP is required")
        if service_port is None:
            raise ValueError("Service port is required")
        try:
            port_value = int(service_port)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid service port: {service_port}") from exc
        return str(name), str(ip), port_value

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def serve_forever(self) -> None:
        if self._started:
            return
        if self._server is None:
            self._server = make_server(self.host, self.port, self.app)
            self._context = self.app.app_context()
        self._started = True
        assert self._context is not None
        self._context.push()
        try:
            assert self._server is not None
            self._server.serve_forever()
        finally:
            self._context.pop()
            self._started = False

    def shutdown(self) -> None:
        if not self._started or self._server is None:
            return
        self._server.shutdown()
