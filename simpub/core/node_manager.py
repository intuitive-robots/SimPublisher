from __future__ import annotations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable
import asyncio
from asyncio import sleep as async_sleep
import socket
import struct
import zmq
import zmq.asyncio
import time
from json import loads
import traceback

from .log import logger
from .utils import (
    send_string_request_async,
    MULTICAST_GRP,
    DISCOVERY_PORT,
    XRNodeInfo,
)


# NOTE: asyncio.loop.sock_recvfrom can only be used after Python 3.11
# So we create a custom DatagramProtocol for multicast discovery
class MulticastDiscoveryProtocol(asyncio.DatagramProtocol):
    """DatagramProtocol for handling multicast discovery messages"""

    def __init__(self, node_manager: XRNodeManager):
        self.node_manager = node_manager
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport
        logger.info("Multicast discovery connection established")

    def datagram_received(self, data: bytes, addr: tuple[str, int]):
        """Handle incoming multicast discovery messages"""
        try:
            node_ip, message = addr[0], data.decode("utf-8")
            node_id, service_port = message[:36], message[36:]
            if node_id not in self.node_manager.xr_nodes_info:
                # Schedule the async registration
                self.node_manager.submit_asyncio_task(
                    self.node_manager.async_register_node_info,
                    node_id,
                    node_ip,
                    service_port,
                )
            self.node_manager.xr_node_heartbeat[node_id] = time.time()
        except Exception as e:
            logger.error(f"Error processing datagram: {e}")
            traceback.print_exc()

    def error_received(self, exc):
        logger.error(f"Multicast protocol error: {exc}")

    def connection_lost(self, exc):
        if exc:
            logger.error(f"Multicast connection lost: {exc}")
        else:
            logger.error("Multicast discovery connection closed")


class XRNodeManager:
    manager: Optional[XRNodeManager] = None

    def __init__(self, host_ip: str) -> None:
        XRNodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context.instance()
        self.host_ip: str = host_ip
        self.xr_nodes_info: Dict[str, XRNodeInfo] = {}
        self.xr_node_heartbeat: Dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.server_future = self.executor.submit(self.thread_task)
        self.discovery_transport = None  # Track the transport for cleanup
        # Wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)

    def start_discover_node_loop(self):
        """Start the async discovery loop in the event loop"""
        if self.loop and self.loop.is_running():
            # Submit the async task to the event loop
            self.discovery_task = self.submit_asyncio_task(
                self.xr_node_discover_loop
            )
            self.node_heartbeat_task = self.submit_asyncio_task(
                self.check_node_heartbeat
            )
        else:
            logger.error(
                "Event loop is not running, cannot start discovery loop"
            )

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def thread_task(self):
        logger.info("The node is running...")
        try:
            self.start_event_loop()
        except KeyboardInterrupt:
            self.stop_node()
        except Exception as e:
            logger.error(f"Unexpected error in thread_task: {e}")
        finally:
            logger.info("The node has been stopped")

    def stop_node(self):
        logger.info("Start to stop the node")
        self.running = False
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.stop_tasks)
                time.sleep(0.1)
                self.loop.stop()
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        logger.info("Start to shutdown the executor")
        self.executor.shutdown(wait=False)
        logger.info("The executor has been shutdown")
        XRNodeManager.manager = None

    def stop_tasks(self):
        # Close discovery transport if it exists
        if self.discovery_transport:
            self.discovery_transport.close()

        # Cancel all running tasks including discovery
        for task in asyncio.all_tasks():
            if task is asyncio.current_task():
                continue
            task.cancel()
        logger.info("All tasks have been cancelled")

    def spin(self):
        while True:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        self.stop_node()
        logger.info("The node has been stopped")

    def submit_asyncio_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def xr_node_discover_loop(self):
        # Create multicast socket
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        # Allow reuse of address
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the port
        sock.bind(("", DISCOVERY_PORT))
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(MULTICAST_GRP),
            socket.inet_aton(self.host_ip),
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        logger.info(
            f"Listening for multicast on {MULTICAST_GRP}:{DISCOVERY_PORT}"
            f" from {self.host_ip}"
        )
        # Get event loop and create datagram endpoint
        loop = asyncio.get_event_loop()
        try:
            # Create the datagram endpoint with the protocol
            transport, _ = await loop.create_datagram_endpoint(
                lambda: MulticastDiscoveryProtocol(self), sock=sock
            )
            # Store transport for cleanup
            self.discovery_transport = transport
            # Keep the loop running until stopped
            while self.running:
                await async_sleep(1)  # Keep the coroutine alive
        except asyncio.CancelledError:
            logger.info("Discovery loop cancelled...")
        except Exception as e:
            logger.error(f"Error in discovery loop: {e}")
            traceback.print_exc()
        finally:
            # Clean up
            if "transport" in locals():
                transport.close()
            sock.close()
            logger.info("Multicast discovery loop stopped")

    async def async_register_node_info(
        self, node_id: str, node_ip: str, service_port: str
    ) -> None:
        try:
            node_info_bytes = await send_string_request_async(
                ["GetNodeInfo", ""], f"tcp://{node_ip}:{service_port}"
            )
            if node_info_bytes is None:
                logger.error(
                    f"Failed to get node info from {node_ip}:{service_port}"
                )
                return
            self.xr_nodes_info[node_id] = loads(
                node_info_bytes.decode("utf-8")
            )
            self.xr_nodes_info[node_id]["ip"] = node_ip
            logger.info(
                f"Registering node info: {self.xr_nodes_info[node_id]['name']}"
                f"/{node_id} at {node_ip}:{service_port}"
            )
        except Exception as e:
            logger.error(f"Error in async_register_node_info: {e}")
            traceback.print_exc()

    async def check_node_heartbeat(self):
        """Check the heartbeat of registered nodes and remove offline nodes."""
        while self.running:
            offline_nodes = [
                node_id
                for node_id, last_heartbeat in self.xr_node_heartbeat.items()
                if time.time() - last_heartbeat > 5.0  # 5 seconds timeout
            ]
            for node_id in offline_nodes:
                logger.warning(
                    f"Node {self.xr_nodes_info[node_id]['name']} "
                    f"{node_id} is offline for 5s, removing it"
                )
                del self.xr_nodes_info[node_id]
                del self.xr_node_heartbeat[node_id]
            await async_sleep(1)


def init_xr_node_manager(ip_addr: Optional[str] = None) -> XRNodeManager:
    if XRNodeManager.manager is not None:
        if ip_addr is None:
            return XRNodeManager.manager
        else:
            raise RuntimeError(
                "XRNodeManager is already initialized, "
                "cannot reinitialize with a different IP address."
            )
    else:
        if ip_addr is None:
            raise ValueError(
                "IP address must be provided for the first initialization"
            )
        logger.info(f"Initializing XRNodeManager with IP {ip_addr}")
        return XRNodeManager(ip_addr)
