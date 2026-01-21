import sys
import time
import threading
import argparse
from pathlib import Path
import inspect
import json
import yaml
import numpy as np
import cv2

# Point cloud processing steps
from ProcessingStep import build_default_steps
from net.PacketV2 import PacketV2Writer
from Actions import PREVIEW, _resolve_colormap_code, _colorize_depth_mm

# SimPublisher core
from simpub.core.simpub_server import ServerBase
from simpub.core.net_component import Streamer as BaseStreamer
from simpub.core.utils import get_zmq_socket_url, send_request_with_addr_async
from simpub.parser.simdata import SimScene, SimObject

# Camera strategies (always required)
from Source import (
    CameraContext,
    AzureKinectCameraStrategy,
    LuxonisCameraStrategy,
    DummyCameraStrategy,
)
# ==================== Pose loading =====================
def load_pose_4x4(pose_path: str | None, align_path: str | None = None) -> np.ndarray | None:
    """
    Load a 4x4 cam_to_world matrix from pose_path.
    If align_path is given and exists, apply:

        T_final = A_align @ T_base
    """
    if not pose_path:
        return None

    p = Path(pose_path)
    if not p.exists():
        print(f"[Pose] Not found: {p}")
        return None

    T = None

    try:
        # ---- 1) Load base cam_to_world into T ----
        if p.suffix.lower() == ".npz":
            data = np.load(p)
            if "cam_to_world" in data and data["cam_to_world"].shape == (4, 4):
                T = data["cam_to_world"].astype(np.float32)
                print(f"[Pose] {p.name}: loaded base cam_to_world.")
            else:
                print(f"[Pose] {p.name}: no 4x4 'cam_to_world' in NPZ.")
                return None
        else:
            # txt / npy fallback
            try:
                T = np.loadtxt(p, dtype=np.float32).reshape(4, 4)
                print(f"[Pose] {p.name}: loaded 4x4 from text.")
            except Exception:
                T = np.load(p).astype(np.float32).reshape(4, 4)
                print(f"[Pose] {p.name}: loaded 4x4 from npy.")


        if T is None:
            print(f"[Pose] {p.name}: failed to load base pose.")
            return None

        # ---- 2) If an alignment matrix exists, multiply it with T
        if align_path:
            ap = Path(align_path)
            if ap.exists():
                try:
                    A = np.load(ap).astype(np.float32)
                    if A.shape == (4, 4):
                        T = A @ T
                        print(f"[Pose] Applied alignment {ap.name} to {p.name}.")
                    else:
                        print(f"[Pose] {ap.name}: not 4x4, ignoring alignment.")
                except Exception as e:
                    print(f"[Pose] Failed to load alignment {ap}: {e}")
            else:
                print(f"[Pose] Alignment file not found: {ap}, ignoring.")

        return T

    except Exception as e:
        print(f"[Pose] Failed to load {p}: {e}")
        return None



# ================ Coordinate conversion =================
def pose_to_unity_coords(T_cam_to_cal: np.ndarray | None) -> np.ndarray | None:
    """Convert calibration/world pose into Unity/IRIS coords."""
    if T_cam_to_cal is None:
        return None

    S_Unity = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # camera frame Y flip to match unity
    F_y = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    T = T_cam_to_cal.astype(np.float32)

    T_cam_to_cal_flipped = T @ F_y

    T_unity = S_Unity @ T_cam_to_cal_flipped
    
    # return (S_Unity @ T_cam_to_cal).astype(np.float32)
    return T_unity.astype(np.float32)


# ================= Intrinsics helpers ==================
def _intr_get(i):
    if i is None:
        return None
    try:
        return float(i.fx), float(i.fy), float(i.cx), float(i.cy), "obj"
    except AttributeError:
        return (
            float(i["fx"]),
            float(i["fy"]),
            float(i["cx"]),
            float(i["cy"]),
            "dict",
        )


def _intr_set(i, fx, fy, cx, cy, kind):
    if i is None:
        return None
    if kind == "obj":
        i.fx, i.fy, i.cx, i.cy = fx, fy, cx, cy
    else:
        i["fx"], i["fy"], i["cx"], i["cy"] = fx, fy, cx, cy
    return i


def _adjust_intrinsics_for_roi_stride(intr, global_cfg):
    tup = _intr_get(intr)
    if tup is None:
        return intr

    fx, fy, cx, cy, kind = tup
    roi = global_cfg.get("roi_xywh")
    stride = int(global_cfg.get("downsample_stride", 1) or 1)

    if roi:
        x0, y0 = int(roi[0]), int(roi[1])
        cx -= x0
        cy -= y0

    if stride > 1:
        fx /= stride
        fy /= stride
        cx /= stride
        cy /= stride

    return _intr_set(intr, fx, fy, cx, cy, kind)


# ================= Camera strategies ===================
def _safe_construct(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**allowed) if allowed else cls()


def make_strategy(cam_cfg: dict, global_cfg: dict):
    """Build the capture strategy."""
    cam_type = str(cam_cfg.get("type", "azure")).lower()
    color_res = tuple(cam_cfg.get("color_res", [1280, 720]))
    align_to_color = bool(global_cfg.get("align_to_color", True))
    usb2mode = bool(cam_cfg.get("usb2mode", False))

    if cam_type in ("azure", "k4a"):
        cam_id = cam_cfg.get("id", "?")
        print(f"[Cam{cam_id}] Using AzureKinectCameraStrategy (type={cam_type})")
        w, h = color_res
        return _safe_construct(
            AzureKinectCameraStrategy,
            width=w,
            height=h,
            color_res=color_res,
            device_index=cam_cfg.get("device", 0),
            align_to_color=align_to_color,
        )

    if cam_type in ("oak", "oakd", "luxonis"):
        w, h = color_res
        return _safe_construct(
            LuxonisCameraStrategy,
            width=w,
            height=h,
            color_res=color_res,
            mxid=cam_cfg.get("mxid"),
            device_index=cam_cfg.get("device", 0),
            usb2mode=usb2mode,
        )

    if cam_type in ("dummy", "sim", "test"):
        w, h = color_res
        return _safe_construct(
            DummyCameraStrategy,
            width=w,
            height=h,
            color_res=color_res,
            fov_deg=cam_cfg.get("fov_deg", 70.0),
            pattern=cam_cfg.get("pattern", "checker"),
            z_mm=cam_cfg.get("z_mm", 1500),
            amp_mm=cam_cfg.get("amp_mm", 250),
            period_s=cam_cfg.get("period_s", 4.0),
            seed=cam_cfg.get("seed"),
        )

    raise ValueError(f"Unknown camera type: {cam_type}")


# ================== Shared frame cache =================
FRAME_CACHE: dict[int, dict] = {}
FRAME_LOCK = threading.Lock()

 
# ============== Camera pipeline thread =================
class CameraPipeline(threading.Thread):
    def __init__(
        self,
        cam_cfg: dict,
        global_cfg: dict,
        preview_rgb: bool,
        preview_depth: bool,
        depth_min_mm: int,
        depth_max_mm: int,
        depth_cmap_code: int,
    ):
        super().__init__(daemon=True)
        self._running = True

        self.cam_id = int(cam_cfg["id"])
        self.global_cfg = global_cfg

        self.fps_max = int(global_cfg.get("fps_max", 30))
        self.frame_period = 1.0 / self.fps_max if self.fps_max > 0 else 0.0

        self.preview_rgb = bool(preview_rgb)
        self.preview_depth = bool(preview_depth)
        self.depth_min_mm = int(depth_min_mm)
        self.depth_max_mm = int(depth_max_mm)
        self.depth_cmap_code = int(depth_cmap_code)

        # Pose (extrinsics + optional alignment -> Unity coords)
        pose_raw = None
        pose_file = cam_cfg.get("pose_file")
        if pose_file:
            pose_dir = Path(global_cfg.get("pose_dir", "."))
            p = Path(pose_file)
            if not p.is_absolute():
                p = pose_dir / p

            align_path = None
            align_file = cam_cfg.get("align_file")
            if align_file:
                ap = Path(align_file)
                if not ap.is_absolute():
                    ap = pose_dir / ap
                align_path = str(ap)

            pose_raw = load_pose_4x4(str(p), align_path)

        # Convert to Unity coords (or identity)
        self.pose = pose_to_unity_coords(pose_raw)
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float32)

        # Camera strategy
        self.strategy = make_strategy(cam_cfg, global_cfg)

        self.ctx = CameraContext(self.strategy)
        self._grab = self.ctx.get_frame
        self._open = (
            getattr(self.ctx, "init", None)
            or getattr(self.ctx, "connect", None)
            or getattr(self.ctx, "open", None)
        )
        self._close = getattr(self.ctx, "close", lambda: None)

        self.steps = build_default_steps(global_cfg)
        self._next_deadline = time.time()

    def run(self):
        try:
            if self._open:
                self._open()
        except Exception as e:
            print(f"[Cam{self.cam_id}] FAILED to open device: {e}")
            return

        try:
            while self._running:
                if not self._grab:
                    break

                tup = self._grab()
                if tup is None:
                    continue

                if isinstance(tup, tuple) and len(tup) == 3:
                    rgb, depth, cfg = tup

                    # Processing steps
                    for s in self.steps:
                        try:
                            rgb, depth = s.process(rgb, depth)
                        except Exception:
                            pass

                    # Adjust intrinsics
                    try:
                        cfg = _adjust_intrinsics_for_roi_stride(
                            cfg, self.global_cfg
                        )
                    except Exception:
                        pass

                    # Preview
                    depth_bgr = None
                    if self.preview_depth and depth is not None:
                        depth_bgr = _colorize_depth_mm(
                            depth,
                            self.depth_min_mm,
                            self.depth_max_mm,
                            self.depth_cmap_code,
                        )
                    if self.preview_rgb:
                        PREVIEW.update(self.cam_id, rgb, depth_bgr)

                    # Cache
                    ts_us = int(time.time() * 1e6)
                    with FRAME_LOCK:
                        FRAME_CACHE[self.cam_id] = {
                            "bgr": rgb,
                            "depth_u16": depth,
                            "intrinsics": np.array(
                                [cfg.fx, cfg.fy, cfg.cx, cfg.cy],
                                dtype=np.float32,
                            ),
                            "timestamp_us": ts_us,
                            "width": rgb.shape[1],
                            "height": rgb.shape[0],
                            "pose_Twc": self.pose,
                        }
                else:
                    # Custom producer path directly populating cache
                    with FRAME_LOCK:
                        FRAME_CACHE[self.cam_id] = tup

                # FPS / throttling
                if self.frame_period > 0:
                    self._next_deadline += self.frame_period
                    now = time.time()
                    if self._next_deadline > now:
                        time.sleep(self._next_deadline - now)
                    else:
                        self._next_deadline = now

        finally:
            try:
                self._close()
            except Exception:
                pass

    def stop(self):
        self._running = False

# ============== RGBD Streamer (SimPublisher wrapper) ==============
class RGBDStreamer(BaseStreamer):
    def __init__(
        self,
        topic_name: str,
        cam_id: int,
        fps: float,
        send_intrinsics: bool = True,
        jpeg_quality: int = 80,
    ):
        self.cam_id = int(cam_id)
        self.send_intrinsics = bool(send_intrinsics)
        self.jpeg_quality = int(jpeg_quality)
        self.packer = PacketV2Writer(send_intrinsics=self.send_intrinsics)

        super().__init__(
            topic_name=topic_name,
            update_func=self.get_update_bytes,
            fps=float(fps),
            start_streaming=False,
        )

        try:
            endpoint = get_zmq_socket_url(self.socket)
            print(f"[RGBD] {topic_name} publishing on {endpoint}")
        except Exception:
            pass

    def get_update_bytes(self) -> bytes:
        with FRAME_LOCK:
            f = FRAME_CACHE.get(self.cam_id)

        if not f:
            return b""

        bgr = f.get("bgr")
        depth = f.get("depth_u16")
        intr = f.get("intrinsics") if self.send_intrinsics else None
        pose_Twc = f.get("pose_Twc")
        ts = int(f.get("timestamp_us", time.time() * 1e6))
        w = int(f.get("width", 0))
        h = int(f.get("height", 0))

        if bgr is not None:
            ok, enc = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            rgb_jpeg = enc.tobytes() if ok else b""
        else:
            rgb_jpeg = b""

        return self.packer.pack(
            camera_id=self.cam_id,
            timestamp_us=ts,
            width=w,
            height=h,
            rgb_jpeg_bytes=rgb_jpeg,
            depth_u16=depth,
            intrinsics=intr,
            pose_Twc=pose_Twc,
        )


# ============== SimPublisher ==============
class PointCloudSimPublisher(ServerBase):
    def __init__(
        self,
        cfg_path: str,
        ip_addr: str,
        preview_rgb_flag: bool,
        preview_depth_flag: bool,
        depth_min_flag: int | None,
        depth_max_flag: int | None,
        depth_cmap_flag: str | None,
    ):
        self.cfg_path = cfg_path
        self.cli_preview_rgb = preview_rgb_flag
        self.cli_preview_depth = preview_depth_flag
        self.cli_depth_min = depth_min_flag
        self.cli_depth_max = depth_max_flag
        self.cli_depth_cmap = depth_cmap_flag

        self.sim_scene = SimScene()
        self.sim_scene.name = "SimScene"  # <- must match Unity side
        self.sim_scene.root = SimObject(name="Root")

        self.workers: list[CameraPipeline] = []
        self.streamers: list[RGBDStreamer] = []

        super().__init__(ip_addr)

    def initialize(self):
        cfg_path = Path(self.cfg_path)
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}")
            sys.exit(1)

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        global_cfg = cfg.get("global", {})
        cameras = cfg.get("cameras", [])
        if not cameras:
            print("No cameras in config.")
            sys.exit(1)

        # Resolve preview flags (CLI overrides YAML)
        preview_rgb = self.cli_preview_rgb or bool(global_cfg.get("preview", False))
        preview_depth = self.cli_preview_depth or bool(
            global_cfg.get("preview_depth", False)
        )

        # Resolve depth range
        dmin = (
            int(self.cli_depth_min)
            if self.cli_depth_min is not None
            else int(global_cfg.get("depth_min_mm", 400))
        )
        dmax = (
            int(self.cli_depth_max)
            if self.cli_depth_max is not None
            else int(global_cfg.get("depth_max_mm", 6000))
        )

        # Resolve colormap
        cmap_name_or_code = (
            self.cli_depth_cmap
            if self.cli_depth_cmap is not None
            else global_cfg.get("depth_colormap", "JET")
        )
        depth_cmap_code = _resolve_colormap_code(cmap_name_or_code)

        if preview_rgb or preview_depth:
            PREVIEW.enable(preview_rgb, preview_depth)

        # Start camera pipelines
        print("=== PointCloudSimPublisher: starting camera pipelines ===")
        for c in cameras:
            cam_id = int(c["id"])
            print(
                f"  - Cam{cam_id}: type={c.get('type', 'azure')} "
                f"topic=RGBD/Cam{cam_id} pose={'yes' if c.get('pose_file') else 'no'}"
            )
            w = CameraPipeline(
                cam_cfg=c,
                global_cfg=global_cfg,
                preview_rgb=preview_rgb,
                preview_depth=preview_depth,
                depth_min_mm=dmin,
                depth_max_mm=dmax,
                depth_cmap_code=depth_cmap_code,
            )
            self.workers.append(w)
            w.start()

        # Create one RGBDStreamer per camera
        fps = float(global_cfg.get("fps_max", 15))
        send_intr = bool(global_cfg.get("send_intrinsics", True))
        jpeg_q = int(global_cfg.get("jpeg_quality", 80))

        for c in cameras:
            cam_id = int(c["id"])
            topic = f"RGBD/Cam{cam_id}"
            s = RGBDStreamer(
                topic_name=topic,
                cam_id=cam_id,
                fps=fps,
                send_intrinsics=send_intr,
                jpeg_quality=jpeg_q,
            )
            s.start_streaming()
            self.streamers.append(s)
            print(f"[SimPub] Streaming '{topic}' at {fps:.1f} Hz")

        # Background task: announce streams to XR nodes
        self.node_manager.submit_asyncio_task(self._auto_subscribe_loop)

    async def _auto_subscribe_loop(self):
        from asyncio import sleep as asyncio_sleep

        announced: set[tuple[str, int]] = set()  # (node_id, cam_id) already subscribed
        spawned: set[str] = set()  # node_ids that already got SpawnSimScene

        while True:
            try:
                registry = getattr(self.node_manager, "xr_nodes", None)
                infos = registry.registered_infos() if registry is not None else []

                for info in infos:
                    services = info.get("serviceList", []) or info.get("services", [])
                    node_id = info.get("nodeID")
                    ip = info.get("ip")
                    port = info.get("port")
                    if not node_id or not ip or not port:
                        continue

                    addr = f"tcp://{ip}:{int(port)}"

                    # --- 1) SpawnSimScene first (once per node) ---
                    if "SpawnSimScene" in services and node_id not in spawned:
                        try:
                            # Delete previous scene if any
                            await send_request_with_addr_async(
                                [b"DeleteSimScene", self.sim_scene.name.encode("utf-8")],
                                addr,
                            )
                        except Exception:
                            # ignore if no scene existed yet
                            pass

                        try:
                            scene_payload = self.sim_scene.serialize().encode("utf-8")
                            reply = await send_request_with_addr_async(
                                [b"SpawnSimScene", scene_payload],
                                addr,
                            )
                            print(
                                f"[SimPub] SpawnSimScene on node "
                                f"{info.get('name', node_id)}: {reply}"
                            )
                            spawned.add(node_id)
                        except Exception as e:
                            print(f"[SimPub] Failed to call SpawnSimScene on {addr}: {e}")
                            # if spawn failed, don't try SubscribePointCloudStream yet
                            continue

                    # --- 2) After scene is spawned, subscribe point cloud streams ---
                    if "SubscribePointCloudStream" not in services:
                        continue

                    for s in self.streamers:
                        key = (node_id, s.cam_id)
                        if key in announced:
                            continue

                        try:
                            url = get_zmq_socket_url(s.socket)
                        except Exception:
                            continue

                        payload = {
                            "camId": s.cam_id,
                            "topic": s.topic_name,
                            "url": url,
                        }
                        msg = [
                            b"SubscribePointCloudStream",
                            json.dumps(payload).encode("utf-8"),
                        ]

                        try:
                            rep = await send_request_with_addr_async(msg, addr)
                            print(
                                f"[SimPub] Announced {s.topic_name} ({url}) "
                                f"to node {info.get('name', node_id)}: {rep}"
                            )
                            announced.add(key)
                        except Exception as e:
                            print(f"[SimPub] Failed to announce to {addr}: {e}")

            except Exception as e:
                print(f"[SimPub] auto_subscribe_loop error: {e}")

            await asyncio_sleep(1.0)

    def shutdown(self):
        # Make shutdown safe to call multiple times
        if getattr(self, "_is_shutdown", False):
            return
        self._is_shutdown = True

        print("[SimPub] Shutting down streamers and cameras...")

        # Stop camera threads
        for w in self.workers:
            try:
                w.stop()
            except Exception:
                pass

        for w in self.workers:
            try:
                w.join(timeout=1.0)
            except Exception:
                pass

        # Stop preview thread / windows
        try:
            if hasattr(PREVIEW, "stop"):
                PREVIEW.stop()
        except Exception:
            pass

        # Extra safety: close all OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Let ServerBase shut down XRNodeManager / executor
        try:
            super().shutdown()
        except Exception:
            pass

        print("[SimPub] Shutdown complete.")


# ======================= CLI + main =======================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/multicam.yaml",
        help="YAML config file for cameras.",
    )
    ap.add_argument(
        "--ip",
        type=str,
        default="10.2.0.2",
        help="IP address to bind SimPublisher server (used by XRNodeManager).",
    )
    # Preview / depth flags (override YAML)
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Enable RGB preview.",
    )
    ap.add_argument(
        "--preview-depth",
        action="store_true",
        help="Enable depth preview.",
    )
    ap.add_argument(
        "--depth-min",
        type=int,
        default=None,
        help="Override depth min (mm).",
    )
    ap.add_argument(
        "--depth-max",
        type=int,
        default=None,
        help="Override depth max (mm).",
    )
    ap.add_argument(
        "--depth-cmap",
        type=str,
        default=None,
        help="Override depth colormap (e.g. JET, TURBO).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    server = PointCloudSimPublisher(
        cfg_path=args.config,
        ip_addr=args.ip,
        preview_rgb_flag=args.preview,
        preview_depth_flag=args.preview_depth,
        depth_min_flag=args.depth_min,
        depth_max_flag=args.depth_max,
        depth_cmap_flag=args.depth_cmap,
    )

    try:
        server.spin()
    except KeyboardInterrupt:
        print("\n[SimPub] KeyboardInterrupt - stopping.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()

# python .\Streamer.py --preview --preview-depth --depth-min 400 --depth-max 4000 --depth-cmap TURBO
