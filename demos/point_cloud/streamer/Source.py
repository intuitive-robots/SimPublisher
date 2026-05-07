from abc import ABC, abstractmethod
import cv2
import zmq
import struct
import socket
import numpy as np
import Datasources as ds
import time
import math
import depthai as dai


from pyk4a import PyK4A, Config, CalibrationType, ColorResolution, DepthMode
from pyk4a.calibration import Calibration

class Source(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def close(self):
        pass



class CameraStrategy(Source):
    @abstractmethod
    def apply_filters(self, depth_frame):
        pass

    @abstractmethod
    def get_intrinsics(self):
        pass



try:
    import depthai as dai
except ImportError:
    dai = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

class RealSenseCameraStrategy(CameraStrategy):
    def __init__(self, width=640, height=480):
        if rs is None:
            raise RuntimeError("RealSense SDK is not installed.")
        self.width = width
        self.height = height
        self.pipeline = None
        self.profile = None

    def connect(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        #depth_frame = self.apply_filters(depth_frame)

        o = self.get_intrinsics()
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data()), ds.CameraConfig(o["fx"], o["fy"], o["ppx"], o["ppy"])

    def apply_filters(self, depth_frame):
        #filtered = self.spatial.process(depth_frame)
        #filtered = self.temporal.process(filtered)
        #filtered = self.hole_filling.process(filtered)
        pass

    def get_intrinsics(self):
        if self.profile is None:
            raise RuntimeError("Camera not connected.")

        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "distortion_coeffs": list(intr.coeffs)
        }

    def close(self):
        self.pipeline.stop()

class LuxonisCameraStrategy:
    """
    Multi-device OAK-D strategy.
    Returns (bgr, depth_u16_mm, cfg) where cfg has: fx, fy, cx, cy, timestamp_us.
    Depth is aligned to RGB and sized exactly to (width, height).
    """

    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 color_res=None,               # optional [W,H] from YAML
                 mxid: str | None = None,      # select device by MXID
                 device_index: int = 0,        # fallback to index if no MXID
                 usb2mode: bool = False,       # diagnostic: force USB2 if needed
                 fps: int = 30,
                 align_to_color: bool = True,  # kept for API symmetry
                 # Quality/robustness knobs (sane defaults)
                 enable_lrc: bool = True,      # REQUIRED when aligning depth to RGB/CENTER on FW 1.2.x
                 enable_subpixel: bool = True,
                 enable_extended: bool = False,
                 confidence: int = 180,        # 
                 median_kernel: str = "KERNEL_5x5",  # "OFF", "KERNEL_3x3", "KERNEL_5x5", "KERNEL_7x7"
                 **_):
        if color_res and len(color_res) == 2:
            width, height = int(color_res[0]), int(color_res[1])

        self.width  = int(width)
        self.height = int(height)
        self.fps    = int(fps)

        self.mxid = mxid
        self.device_index = int(device_index)
        self.usb2mode = bool(usb2mode)

        self.enable_lrc = bool(enable_lrc)
        self.enable_subpixel = bool(enable_subpixel)
        self.enable_extended = bool(enable_extended)
        self.confidence = int(max(0, min(255, confidence)))
        self.median_kernel = str(median_kernel).upper()

        self.dev = None
        self.qRgb = None
        self.qDepth = None
        self._intr = None  # np.array([fx,fy,cx,cy], float32)

    # API aliases for your CameraContext
    def open(self):       return self._connect()
    def init(self):       return self._connect()
    def connect(self):    return self._connect()
    def disconnect(self): return self.close()

    def _connect(self):
        # Enumerate & print devices (helps catch wrong MXID)
        devs = dai.Device.getAllAvailableDevices()
        found = [d.getMxId() for d in devs]
        print(f"[OAK] Available devices: {found if found else '[]'}")
        if self.mxid and self.mxid not in found:
            raise RuntimeError(f"MXID '{self.mxid}' not found. Detected: {found}")

        # ------- Build pipeline -------
        p = dai.Pipeline()

        # Color camera (RGB)
        camRgb = p.create(dai.node.ColorCamera)
        rgb_sock = getattr(dai.CameraBoardSocket, "RGB",
                           getattr(dai.CameraBoardSocket, "CAM_A"))
        camRgb.setBoardSocket(rgb_sock)
        camRgb.setPreviewSize(self.width, self.height)
        camRgb.setFps(self.fps)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Stereo inputs (mono L/R) — OV7251 supports 400p/480p only
        monoL = p.create(dai.node.MonoCamera)
        monoR = p.create(dai.node.MonoCamera)
        monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoL.setFps(self.fps)
        monoR.setFps(self.fps)

        # Stereo depth (quality + stability)
        stereo = p.create(dai.node.StereoDepth)
        try:
            PM = dai.node.StereoDepth.PresetMode   # modern path
        except AttributeError:
            PM = dai.StereoDepth.PresetMode
        # Version-safe dense preset
        if hasattr(PM, "HIGH_DENSITY"):
            preset = PM.HIGH_DENSITY
        elif hasattr(PM, "MEDIUM_DENSITY"):
            preset = PM.MEDIUM_DENSITY
        elif hasattr(PM, "DEFAULT"):
            preset = PM.DEFAULT
        else:
            # last resort: fall back to any attr that exists
            preset = [v for k,v in PM.__dict__.items() if k.isupper()][0]
        stereo.setDefaultProfilePreset(preset)

        # Confidence (cleaner, fewer speckles)
        stereo.initialConfig.setConfidenceThreshold(self.confidence)

        cfg = stereo.initialConfig.get()

        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 20

        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 2

        cfg.postProcessing.temporalFilter.enable = True

        stereo.initialConfig.set(cfg)

        # Median filter (small kernel smooths speckle nicely)
        kernel_map = {
            "OFF":        dai.MedianFilter.MEDIAN_OFF,
            "KERNEL_3X3": dai.MedianFilter.KERNEL_3x3,
            "KERNEL_5X5": dai.MedianFilter.KERNEL_5x5,
            "KERNEL_7X7": dai.MedianFilter.KERNEL_7x7,
        }
        stereo.setMedianFilter(kernel_map.get(self.median_kernel, dai.MedianFilter.KERNEL_5x5))

        # Align to RGB and match sizes exactly.
        # NOTE: On FW 1.2.x, CENTER/RGB alignment REQUIRES LRC to be enabled.
        stereo.setDepthAlign(rgb_sock)
        stereo.setLeftRightCheck(True)

        # Keep disparity boosters off (reduce artifacts / avoid old warnings)
        stereo.setExtendedDisparity(bool(self.enable_extended))
        stereo.setSubpixel(bool(self.enable_subpixel))

        stereo.setOutputSize(self.width, self.height)

        # Links
        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)

        xoutRgb   = p.create(dai.node.XLinkOut); xoutRgb.setStreamName("rgb")
        xoutDepth = p.create(dai.node.XLinkOut); xoutDepth.setStreamName("depth")
        camRgb.preview.link(xoutRgb.input)
        stereo.depth.link(xoutDepth.input)

        # ------- Open specific device (critical for multi-camera) -------
        if self.mxid:
            info = dai.DeviceInfo(self.mxid)
            self.dev = dai.Device(p, info, usb2Mode=self.usb2mode)
            print(f"[OAK] Opening device MXID={self.mxid} usb2={self.usb2mode}")
        else:
            if not devs:
                raise RuntimeError("No OAK devices found.")
            info = devs[min(self.device_index, len(devs)-1)]
            self.dev = dai.Device(p, info, usb2Mode=self.usb2mode)
            print(f"[OAK] Opening device MXID={info.getMxId()} (index={self.device_index}) usb2={self.usb2mode}")

        # Small, non-blocking queues for low latency
        self.qRgb   = self.dev.getOutputQueue("rgb",   maxSize=1, blocking=False)
        self.qDepth = self.dev.getOutputQueue("depth", maxSize=1, blocking=False)

        # Intrinsics for the RGB camera at (width,height)
        calib = self.dev.readCalibration()
        K = calib.getCameraIntrinsics(rgb_sock, self.width, self.height)
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
        self._intr = np.array([fx, fy, cx, cy], dtype=np.float32)

    def get_frame(self):
        """
        Returns (bgr, depth_u16_mm, cfg). Never returns None; if packets are late,
        it blocks briefly and falls back gracefully.
        """
        if self.qRgb is None or self.qDepth is None:
            time.sleep(0.005)
            return self._blank_frame()

        try:
            # Try-get with a short deadline to keep UI responsive
            deadline = time.time() + 0.20
            rgb_pkt = None
            d_pkt   = None
            while time.time() < deadline and (rgb_pkt is None or d_pkt is None):
                if rgb_pkt is None:
                    rgb_pkt = self.qRgb.tryGet()
                if d_pkt is None:
                    d_pkt   = self.qDepth.tryGet()
                if rgb_pkt is None or d_pkt is None:
                    time.sleep(0.001)

            if rgb_pkt is None:
                rgb_pkt = self.qRgb.get()     # block if still missing
            if d_pkt is None:
                d_pkt = self.qDepth.get()     # block if still missing

            bgr   = rgb_pkt.getCvFrame()                 # HxWx3 uint8 (BGR)
            depth = d_pkt.getFrame().copy()              # HxW uint16 (mm)

            if depth.shape[:2] != bgr.shape[:2]:
                depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            class _Cfg: pass
            cfg = _Cfg()
            cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
            cfg.timestamp_us = int(time.time() * 1e6)
            return bgr, depth.astype(np.uint16), cfg

        except Exception as e:
            # Device glitch or link error; return a blank frame so the pipeline keeps running
            print(f"[OAK] queue error: {e}")
            return self._blank_frame()

    def _blank_frame(self):
        # Minimal fallback frame to keep upstream safe
        bgr = np.zeros((self.height, self.width, 3), np.uint8)
        depth = np.zeros((self.height, self.width), np.uint16)
        class _Cfg: pass
        cfg = _Cfg()
        if self._intr is None:
            fx = fy = 1.0; cx = self.width * 0.5; cy = self.height * 0.5
            intr = np.array([fx, fy, cx, cy], dtype=np.float32)
        else:
            intr = self._intr
        cfg.fx, cfg.fy, cfg.cx, cfg.cy = intr
        cfg.timestamp_us = int(time.time() * 1e6)
        return bgr, depth, cfg

    def get_intrinsics(self):
        if self._intr is None:
            return {"fx": 0.0, "fy": 0.0, "ppx": 0.0, "ppy": 0.0, "model": "perspective", "distortion_coeffs": []}
        fx, fy, cx, cy = map(float, self._intr)
        return {"fx": fx, "fy": fy, "ppx": cx, "ppy": cy, "model": "perspective", "distortion_coeffs": []}

    def close(self):
        if self.dev is not None:
            try:
                self.dev.close()
            finally:
                self.dev = None
                self.qRgb = None
                self.qDepth = None


class AzureKinectCameraStrategy(CameraStrategy):
    def __init__(self, width=640, height=480, device_index=0):
        self.width = width
        self.height = height
        self.device_index = int(device_index)   
        self.device = None

    def connect(self):
        self.device = PyK4A(
            Config(
                color_resolution=self._get_color_resolution(),
                depth_mode=self._get_depth_mode(),
                synchronized_images_only=True
            ),
            device_id=self.device_index          
        )
        self.device.start()


    def get_frame(self):
        capture = self.device.get_capture()
        if capture.color is None or capture.depth is None:
            return None, None, None

        color = capture.color
        depth = capture.transformed_depth  # depth aligned to color

    # Ensure 3-channel BGR for JPEG encoding
        if color.ndim == 3 and color.shape[2] == 4:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

        intr = self.get_intrinsics()
        return color, depth, ds.CameraConfig(intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])

    def get_intrinsics(self):
        calib: Calibration = self.device.calibration
        cam = calib.get_camera_matrix(CalibrationType.COLOR)
        fx = cam[0, 0]
        fy = cam[1, 1]
        ppx = cam[0, 2]
        ppy = cam[1, 2]
        return {
            "fx": fx,
            "fy": fy,
            "ppx": ppx,
            "ppy": ppy,
            "matrix": cam.tolist()
        }

    def close(self):
        if self.device is not None:
            self.device.stop()

    def _get_color_resolution(self):
        # Map width x height to Azure Kinect color resolution
        if self.width == 1280 and self.height == 720:
            return ColorResolution.RES_720P
        elif self.width == 1920 and self.height == 1080:
            return ColorResolution.RES_1080P
        else:
            return ColorResolution.RES_720P  # default

    def _get_depth_mode(self):
        return DepthMode.NFOV_2X2BINNED # 640x576 (highest quality narrow FOV)

    def apply_filters(self, depth_frame):
        pass



class CameraContext:
    def __init__(self, strategy: Source):
        self.strategy = strategy
        self.is_camera = isinstance(strategy, CameraStrategy)

    def init(self):
        self.strategy.connect()

    def get_frame(self):
        rgb, depth, config = self.strategy.get_frame()
        return rgb, depth, config

    def apply_filters(self, depth_frame):
        if self.is_camera:
            return self.strategy.apply_filters(depth_frame)
        else:
            print("Filter not available for this source.")
            return depth_frame

    def get_intrinsics(self):
        if self.is_camera:
            return self.strategy.get_intrinsics()
        else:
            return None

    def close(self):
        self.strategy.close()

class DummyCameraStrategy:
    """
    Synthetic camera for testing multi-cam and poses without hardware.

    Output per frame:
      (bgr: HxWx3 uint8, depth_u16: HxW uint16 mm, cfg: object with fx,fy,cx,cy,timestamp_us)

    __init__ signature mirrors other strategies so Streamer._safe_construct can pass
    common kwargs like width/height/color_res/align_to_color.
    """
    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 color_res=None,
                 fov_deg: float = 70.0,
                 pattern: str = "checker",   # "checker" or "bars"
                 z_mm: int = 1500,           # base Z in millimetres
                 amp_mm: int = 250,          # wave amplitude in mm
                 period_s: float = 4.0,      # animation period (seconds)
                 seed: int | None = None,
                 align_to_color: bool = True,  # kept for API symmetry (not used)
                 **_):
        # Resolve output size using color_res if provided
        if color_res is not None and len(color_res) == 2:
            width, height = int(color_res[0]), int(color_res[1])
        self.w, self.h = int(width), int(height)

        self.fov_deg = float(fov_deg)
        self.pattern = str(pattern).lower()
        self.z_base  = int(z_mm)
        self.amp     = int(amp_mm)
        self.period  = float(period_s)

        self.t0      = None
        self.rng     = np.random.RandomState(None if seed is None else int(seed))
        self._intr   = self._compute_intrinsics(self.w, self.h, self.fov_deg)

    # ---------------- API expected by CameraContext ----------------

    def open(self):
        """Open/init the synthetic source."""
        self.t0 = time.time()

    # Some contexts call these names instead; alias to open/close for consistency.
    def init(self):       return self.open()
    def connect(self):    return self.open()
    def close(self):      return None
    def disconnect(self): return self.close()

    # ---------------- Internals ----------------

    @staticmethod
    def _compute_intrinsics(w, h, fov_deg):
        """
        Compute simple pinhole intrinsics with fx=fy from horizontal FOV,
        and principal point at the image center.
        """
        fx = (0.5 * w) / math.tan(math.radians(fov_deg) * 0.5)
        fy = fx
        cx = 0.5 * w
        cy = 0.5 * h
        return np.array([fx, fy, cx, cy], dtype=np.float32)

    def _rgb_frame(self, t):
        """Generate a synthetic RGB frame (uint8 BGR)."""
        if self.pattern == "bars":
            u = (np.arange(self.w, dtype=np.float32)[None, :] / 16.0)
            img = np.zeros((self.h, self.w, 3), np.uint8)
            phase = (t / self.period) * 255.0
            img[..., 0] = np.mod(u * 10.0 + phase, 255).astype(np.uint8)
            img[..., 1] = np.mod(u * 20.0 + phase, 255).astype(np.uint8)
            img[..., 2] = np.mod(u * 30.0 + phase, 255).astype(np.uint8)
            return img

        # default: checker
        u = (np.arange(self.w, dtype=np.float32)[None, :] / 32.0)
        v = (np.arange(self.h, dtype=np.float32)[:, None] / 32.0)
        phase = 2.0 * math.pi * (t / self.period)
        checker = ((np.floor(u + phase) + np.floor(v)) % 2).astype(np.float32)

        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        img[..., 0] = (checker * 220 + 20).astype(np.uint8)                # B
        img[..., 1] = ((1.0 - checker) * 220 + 20).astype(np.uint8)        # G
        img[..., 2] = (40 + 30*np.sin(u*0.5 + v*0.25 + phase)).astype(np.uint8)  # R
        return img

    def _depth_frame(self, t):
        """Generate a synthetic depth map (uint16 millimetres)."""
        xs = np.linspace(-1.0, 1.0, self.w, dtype=np.float32)[None, :]
        ys = np.linspace(-1.0, 1.0, self.h, dtype=np.float32)[:,   None]
        phase = 2.0 * math.pi * (t / self.period)

        z = (self.z_base
             + self.amp * np.sin(xs * 2.0 + phase)
             + 0.5 * self.amp * np.sin(ys * 1.5 - phase * 0.7))

        z[z < 300.0] = 0.0  # invalidate too-close values like a real sensor
        depth = z.astype(np.uint16)

        # Sparse random holes to mimic stereo confidence dropouts
        holes = (self.rng.rand(self.h, self.w) < 0.002)
        depth[holes] = 0
        return depth

    # ---------------- Frame grab ----------------

    def get_frame(self):
        """
        Return a frame tuple like real strategies do:
            (bgr: HxWx3 uint8, depth_u16: HxW uint16, cfg: object with fx,fy,cx,cy,timestamp_us)
        """
        if self.t0 is None:
            self.open()
        t = time.time() - self.t0

        bgr   = self._rgb_frame(t)
        depth = self._depth_frame(t)

        # Pack intrinsics / timestamp in a simple attribute object
        class _Cfg: pass
        cfg = _Cfg()
        cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
        cfg.timestamp_us = int(time.time() * 1e6)

        return bgr, depth, cfg
    """
    Synthetic camera for testing multi-cam and poses without hardware.

    __init__ signature mirrors other strategies so Streamer._safe_construct can
    pass common kwargs like width/height/color_res/align_to_color.
    """
    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 color_res=None,
                 fov_deg: float = 70.0,
                 pattern: str = "checker",
                 z_mm: int = 1500,
                 amp_mm: int = 250,
                 period_s: float = 4.0,
                 seed: int | None = None,
                 align_to_color: bool = True,  # kept for API symmetry
                 **_):
        # Resolve output size (use color_res if provided)
        if color_res is not None and len(color_res) == 2:
            width, height = int(color_res[0]), int(color_res[1])
        self.w, self.h = int(width), int(height)

        self.fov_deg = float(fov_deg)
        self.pattern = str(pattern)
        self.z_base  = int(z_mm)
        self.amp     = int(amp_mm)
        self.period  = float(period_s)
        self.t0      = None
        self.rng     = np.random.RandomState(None if seed is None else int(seed))

        # Precompute intrinsics (fx=fy from horiz. FOV; principal point at center)
        self._intr = self._compute_intrinsics(self.w, self.h, self.fov_deg)

    @staticmethod
    def _compute_intrinsics(w, h, fov_deg):
        import math
        fx = (0.5 * w) / math.tan(math.radians(fov_deg) * 0.5)
        fy = fx
        cx = 0.5 * w
        cy = 0.5 * h
        return np.array([fx, fy, cx, cy], dtype=np.float32)

    # For API symmetry with other strategies
    def open(self):  # or init/connect depending on your CameraContext
        self.t0 = time.time()

    def close(self):
        pass

    def _rgb_frame(self, t):
        # Moving checkerboard with a subtle animated tint
        u = (np.arange(self.w)[None, :] / 32.0)
        v = (np.arange(self.h)[:, None] / 32.0)
        phase = 2 * np.pi * (t / self.period)

        if self.pattern.lower() == "bars":
            img = np.zeros((self.h, self.w, 3), np.uint8)
            img[..., 0] = ((u * 32 + phase) % 255).astype(np.uint8)
            img[..., 1] = ((u * 64 + phase) % 255).astype(np.uint8)
            img[..., 2] = ((u * 96 + phase) % 255).astype(np.uint8)
            return img

        # default: checker
        checker = ((np.floor(u + phase) + np.floor(v)) % 2).astype(np.float32)
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[..., 0] = (checker * 220 + 20).astype(np.uint8)            # B
        img[..., 1] = ((1 - checker) * 220 + 20).astype(np.uint8)      # G
        img[..., 2] = (40 + 30 * np.sin(u * 0.5 + v * 0.25 + phase)).astype(np.uint8)  # R
        return img

    def _depth_frame(self, t):
        # Sinusoidal “wavy plane” in front of camera (millimetres)
        xs = np.linspace(-1, 1, self.w)[None, :]
        ys = np.linspace(-1, 1, self.h)[:, None]
        phase = 2 * np.pi * (t / self.period)
        z = (self.z_base
             + self.amp * np.sin(xs * 2.0 + phase)
             + 0.5 * self.amp * np.sin(ys * 1.5 - phase * 0.7))
        z[z < 300] = 0  # invalidate too-close like real sensors
        depth_u16 = z.astype(np.uint16)
        # Add sparse holes
        mask = (self.rng.rand(self.h, self.w) < 0.002)
        depth_u16[mask] = 0
        return depth_u16

    def get_frame(self):
        if self.t0 is None:
            self.open()
        t = time.time() - self.t0
        bgr = self._rgb_frame(t)
        depth = self._depth_frame(t)

        # Pack intrinsics + timestamp the same way as other strategies
        class _Cfg: pass
        cfg = _Cfg()
        cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
        cfg.timestamp_us = int(time.time() * 1e6)
        return bgr, depth, cfg
