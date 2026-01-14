from __future__ import annotations

import traceback
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request
from werkzeug.serving import BaseWSGIServer, make_server

from ..core.utils import XRNodeRegistry, send_request_with_addr
from .app.utils import read_qr_alignment_data, send_zmq_request

_ROUTES: List[Tuple[str, str, Dict[str, object]]] = []


def route(rule: str, **options: object):
    """Decorator for registering SimPubWebServer routes."""

    def decorator(func):
        _ROUTES.append((func.__name__, rule, dict(options)))
        return func

    return decorator


class SimPubWebServer:
    """Flask server exposing XR node management endpoints."""

    def __init__(
        self,
        xr_nodes: XRNodeRegistry,
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        self.xr_nodes = xr_nodes
        self.host = host
        self.port = port
        root_dir = Path(__file__).resolve().parent / "app"
        template_dir = root_dir / "templates"
        static_dir = root_dir / "static"
        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir),
            static_url_path="/static",
        )
        self.app.config["XR_NODE_REGISTRY"] = xr_nodes
        self._server: Optional[BaseWSGIServer] = None
        self._context = None
        self._started: bool = False

        for handler_name, rule, options in _ROUTES:
            self._register_route(handler_name, rule, options)

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------
    @route("/", methods=["GET"])
    def index(self):
        return render_template("index.html")

    @route("/scan", methods=["GET"])
    def scan(self):
        try:
            nodes = []
            for node in self.xr_nodes.registered_infos():
                node_payload = dict(node)
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

    @route("/teleport-scene", methods=["POST"])
    def teleport_scene(self):
        payload = request.get_json(silent=True) or {}
        # new_name = payload.get("newName")
        # if not new_name:
        #     return (
        #         jsonify(
        #             {
        #                 "status": "error",
        #                 "message": "New name is required",
        #             }
        #         ),
        #         400,
        #     )
        try:
            _, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            response = send_request_with_addr(
                "ToggleGrab", "", f"tcp://{ip}:{service_port}"
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

    @route("/rename-device", methods=["POST"])
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
            response = send_request_with_addr(
                "Rename", new_name, f"tcp://{ip}:{service_port}"
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

    @route("/env-occlusion", methods=["POST"])
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

    @route("/toggle-qr-tracking", methods=["POST"])
    def toggle_qr_tracking(self):
        payload = request.get_json(silent=True) or {}
        try:
            _, ip, service_port = self._extract_connection_info(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        try:
            response = send_request_with_addr(
                "ToggleQRTracking", "", f"tcp://{ip}:{service_port}"
            )
            return jsonify(
                {
                    "status": "success",
                    "message": "Toggle QR Tracking",
                    "response": response,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Error during Toggle QR Tracking: {exc}",
                    }
                ),
                500,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_connection_info(
        self, payload: dict[str, object]
    ) -> Tuple[str, str, int]:
        if not all(k in payload for k in ("name", "ip", "servicePort")):
            raise ValueError("Missing required fields: name, ip, servicePort")
        name = str(payload["name"])
        ip = str(payload["ip"])
        service_port = int(str(payload["servicePort"]))
        return (name, ip, service_port)

    def _register_route(
        self,
        handler_name: str,
        rule: str,
        options: Dict[str, object],
    ) -> None:
        handler = getattr(self, handler_name)

        @wraps(handler)
        def bound_handler(*args, **kwargs):
            return handler(*args, **kwargs)

        self.app.add_url_rule(
            rule,
            endpoint=handler_name,
            view_func=bound_handler,
            provide_automatic_options=False,
            **options,
        )

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
