from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Union, Any

import pyzlc

from .log import logger
from .simpub_server import ServerBase
from .utils import XRNodeInfo


class TrajectoryConfigDict(TypedDict):
    name: str
    points: List[List[float]]
    color: List[float]
    width: float
    resolution: int


class XRTrajectory:
    def __init__(
        self,
        cavns: XRCavns,
        name: str,
        points: List[List[float]],
        color: List[float],
        width: float,
        resolution: int,
    ) -> None:
        self._cavns = cavns
        self.name = name
        self.points = points
        self.color = color
        self.width = width
        self.resolution = resolution
        self.valid = True

    def update(
        self,
        points: Optional[List[List[Union[int, float]]]] = None,
        color: Optional[List[Union[int, float]]] = None,
        width: Optional[Union[int, float]] = None,
        resolution: Optional[int] = None,
    ):
        self._cavns.update_trajectory(
            name=self.name,
            points=points,
            color=color,
            width=width,
            resolution=resolution,
        )
        cfg = self._cavns._trajectories.get(self.name)
        if cfg is not None:
            self.points = cfg["points"]
            self.color = cfg["color"]
            self.width = cfg["width"]
            self.resolution = cfg["resolution"]

    def delete(self) -> None:
        self._cavns.delete_trajectory(self.name)
        self.valid = False


class XRCavns(ServerBase):

    def __init__(self, ip_addr: str = "127.0.0.1") -> None:
        self._trajectories: Dict[str, TrajectoryConfigDict] = {}
        super().__init__(server_name="XRCavns", ip_addr=ip_addr)

    def initialize(self) -> None:
        pass

    async def on_new_device_found(self, xr_info: XRNodeInfo):
        logger.info("New XR device found: %s", xr_info.get("name", "Unknown"))
        for trajectory_name, config in self._trajectories.items():
            try:
                await pyzlc.async_call(
                    f"{xr_info['name']}/SpawnTrajectory",
                    config,
                )
                logger.debug(
                    "Sent existing trajectory '%s' to new device '%s'",
                    trajectory_name,
                    xr_info.get("name", "Unknown"),
                )
            except Exception as e:
                logger.error(
                    "Failed to send trajectory '%s' to device '%s': %s",
                    trajectory_name,
                    xr_info.get("name", "Unknown"),
                    e,
                )
                logger.debug("Exception details:", exc_info=True)
        pyzlc.info(
            "Current trajectories in registry: %s",
            list(self._trajectories.keys()),
        )


    def _validate_trajectory_points(
        self, points: List[List[Union[int, float]]]
    ) -> List[List[float]]:
        if not points:
            raise ValueError("Trajectory points must be a non-empty list.")
        validated: List[List[float]] = []
        for idx, point in enumerate(points):
            if point is None or len(point) < 3:
                raise ValueError(
                    f"Trajectory point at index {idx} must have at least 3 values."
                )
            validated.append([float(point[0]), float(point[1]), float(point[2])])
        return validated

    def _normalize_trajectory_color(
        self, color: Optional[List[Union[int, float]]]
    ) -> List[float]:
        if color is None:
            return [1.0, 1.0, 1.0, 1.0]
        if len(color) == 3:
            return [float(color[0]), float(color[1]), float(color[2]), 1.0]
        if len(color) >= 4:
            return [
                float(color[0]),
                float(color[1]),
                float(color[2]),
                float(color[3]),
            ]
        logger.warning("Invalid trajectory color provided; using default [1,1,1,1].")
        return [1.0, 1.0, 1.0, 1.0]

    def _build_trajectory_config(
        self,
        name: str,
        points: List[List[Union[int, float]]],
        color: Optional[List[Union[int, float]]] = None,
        width: Union[int, float] = 0.01,
        resolution: int = 10,
    ) -> TrajectoryConfigDict:
        normalized_width = float(width) if float(width) > 0 else 0.01
        normalized_resolution = int(resolution) if int(resolution) > 0 else 10
        return {
            "name": name,
            "points": self._validate_trajectory_points(points),
            "color": self._normalize_trajectory_color(color),
            "width": normalized_width,
            "resolution": normalized_resolution,
        }

    def _broadcast_trajectory_call(self, service_name: str, payload: Any):
        for xr_info in pyzlc.get_nodes_info():
            device_name = xr_info.get("name", "")
            try:
                pyzlc.call(f"{device_name}/{service_name}", payload)
            except Exception as exc:
                logger.error(
                    "Trajectory service call failed for %s/%s: %s",
                    device_name,
                    service_name,
                    exc,
                )

    def create_trajectory(
        self,
        name: str,
        points: List[List[Union[int, float]]],
        color: Optional[List[Union[int, float]]] = None,
        width: Union[int, float] = 0.01,
        resolution: int = 10,
    ) -> XRTrajectory:
        config = self._build_trajectory_config(
            name=name,
            points=points,
            color=color,
            width=width,
            resolution=resolution,
        )
        self._broadcast_trajectory_call("SpawnTrajectory", config)
        self._trajectories[name] = config
        return XRTrajectory(
            cavns=self,
            name=name,
            points=config["points"],
            color=config["color"],
            width=config["width"],
            resolution=config["resolution"],
        )

    def update_trajectory(
        self,
        name: str,
        points: Optional[List[List[Union[int, float]]]] = None,
        color: Optional[List[Union[int, float]]] = None,
        width: Optional[Union[int, float]] = None,
        resolution: Optional[int] = None,
    ):
        if name not in self._trajectories:
            pyzlc.warning("Attempted to update non-existent trajectory '%s'", name)

        current = self._trajectories[name]
        next_points = points if points is not None else current["points"]
        next_color: Optional[List[Union[int, float]]] = (
            color if color is not None else current["color"]
        )
        next_width = width if width is not None else current["width"]
        next_resolution = (
            resolution if resolution is not None else current["resolution"]
        )

        config = self._build_trajectory_config(
            name=name,
            points=next_points,
            color=next_color,
            width=next_width,
            resolution=next_resolution,
        )
        self._broadcast_trajectory_call("UpdateTrajectory", config)
        self._trajectories[name] = config

    def delete_trajectory(self, name: str):
        if name not in self._trajectories:
            pyzlc.warning("Attempted to delete non-existent trajectory '%s'", name)
        status = self._broadcast_trajectory_call("DeleteTrajectory", name)
        self._trajectories.pop(name, None)
        return status

    def clear_trajectories(self) -> None:
        for name in list(self._trajectories.keys()):
            self._broadcast_trajectory_call("DeleteTrajectory", name)
        self._trajectories.clear()
