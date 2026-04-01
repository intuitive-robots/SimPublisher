from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Union, Any

import pyzlc

from .log import logger
from .simpub_server import ServerBase
from .utils import XRNodeInfo


class TrajectoryWaypointDict(TypedDict):
    pos: List[float]  # [x, y, z]
    color: List[float]  # [r, g, b, a]


class TrajectoryConfigDict(TypedDict):
    name: str
    waypoints: List[TrajectoryWaypointDict]
    width: float
    resolution: int


class XRTrajectory:
    def __init__(
        self,
        cavns: XRCavns,
        name: str,
        waypoints: List[TrajectoryWaypointDict],
        width: float,
        resolution: int,
    ) -> None:
        self._cavns = cavns
        self.name = name
        self.waypoints = waypoints
        self.width = width
        self.resolution = resolution
        self.valid = True

    def update(
        self,
        waypoints: Optional[List[TrajectoryWaypointDict]] = None,
        points: Optional[List[List[Union[int, float]]]] = None,
        color: Optional[List[Union[int, float]]] = None,
        width: Optional[Union[int, float]] = None,
        resolution: Optional[int] = None,
    ):
        self._cavns.update_trajectory(
            name=self.name,
            waypoints=waypoints,
            points=points,
            color=color,
            width=width,
            resolution=resolution,
        )
        cfg = self._cavns._trajectories.get(self.name)
        if cfg is not None:
            self.waypoints = cfg["waypoints"]
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

    def _normalize_color(self, color: Optional[List[Union[int, float]]]) -> List[float]:
        if color is None:
            return [1.0, 1.0, 1.0, 1.0]
        if len(color) == 3:
            return [float(color[0]), float(color[1]), float(color[2]), 1.0]
        if len(color) >= 4:
            return [float(color[0]), float(color[1]), float(color[2]), float(color[3])]
        logger.warning("Invalid color provided; using default [1,1,1,1].")
        return [1.0, 1.0, 1.0, 1.0]

    def _normalize_position(self, pos: List[Union[int, float]]) -> List[float]:
        if pos is None or len(pos) < 3:
            raise ValueError("Position must have at least 3 values [x, y, z].")
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    def _validate_waypoints(
        self, waypoints: List[TrajectoryWaypointDict]
    ) -> List[TrajectoryWaypointDict]:
        if not waypoints:
            raise ValueError("Waypoints must be a non-empty list.")
        validated: List[TrajectoryWaypointDict] = []
        for idx, wp in enumerate(waypoints):
            if "pos" not in wp:
                raise ValueError(f"Waypoint at index {idx} missing 'pos' field.")
            validated.append(
                {
                    "pos": self._normalize_position(wp["pos"]),
                    "color": self._normalize_color(wp.get("color")),
                }
            )
        return validated

    def _points_to_waypoints(
        self,
        points: List[List[Union[int, float]]],
        color: Optional[List[Union[int, float]]] = None,
    ) -> List[TrajectoryWaypointDict]:
        """Convert legacy points + color format to waypoints format."""
        normalized_color = self._normalize_color(color)
        waypoints: List[TrajectoryWaypointDict] = []
        for point in points:
            waypoints.append(
                {
                    "pos": self._normalize_position(point),
                    "color": normalized_color,
                }
            )
        return waypoints

    def _build_trajectory_config(
        self,
        name: str,
        waypoints: List[TrajectoryWaypointDict],
        width: Union[int, float] = 0.01,
        resolution: int = 10,
    ) -> TrajectoryConfigDict:
        normalized_width = float(width) if float(width) > 0 else 0.01
        normalized_resolution = int(resolution) if int(resolution) > 0 else 10
        return {
            "name": name,
            "waypoints": self._validate_waypoints(waypoints),
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
        waypoints: Optional[List[TrajectoryWaypointDict]] = None,
        points: Optional[List[List[Union[int, float]]]] = None,
        color: Optional[List[Union[int, float]]] = None,
        width: Union[int, float] = 0.01,
        resolution: int = 10,
    ) -> XRTrajectory:
        """Create a trajectory.

        Args:
            name: Trajectory name
            waypoints: List of waypoints with pos and color (new format)
            points: List of [x,y,z] positions (legacy format, use with color)
            color: Default color for all points (legacy format, used with points)
            width: Line width
            resolution: Interpolation segments per control point
        """
        if waypoints is not None:
            final_waypoints = waypoints
        elif points is not None:
            final_waypoints = self._points_to_waypoints(points, color)
        else:
            raise ValueError("Must provide either 'waypoints' or 'points'.")

        config = self._build_trajectory_config(
            name=name,
            waypoints=final_waypoints,
            width=width,
            resolution=resolution,
        )
        self._broadcast_trajectory_call("SpawnTrajectory", config)
        self._trajectories[name] = config
        return XRTrajectory(
            cavns=self,
            name=name,
            waypoints=config["waypoints"],
            width=config["width"],
            resolution=config["resolution"],
        )

    def update_trajectory(
        self,
        name: str,
        waypoints: Optional[List[TrajectoryWaypointDict]] = None,
        points: Optional[List[List[Union[int, float]]]] = None,
        color: Optional[List[Union[int, float]]] = None,
        width: Optional[Union[int, float]] = None,
        resolution: Optional[int] = None,
    ):
        if name not in self._trajectories:
            pyzlc.warning("Attempted to update non-existent trajectory '%s'", name)
            return

        current = self._trajectories[name]

        # Determine waypoints
        if waypoints is not None:
            final_waypoints = waypoints
        elif points is not None:
            final_waypoints = self._points_to_waypoints(points, color)
        else:
            final_waypoints = current["waypoints"]

        next_width = width if width is not None else current["width"]
        next_resolution = (
            resolution if resolution is not None else current["resolution"]
        )

        config = self._build_trajectory_config(
            name=name,
            waypoints=final_waypoints,
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
