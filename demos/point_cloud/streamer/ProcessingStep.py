from abc import ABC, abstractmethod
import numpy as np
from skimage.measure import block_reduce
import cv2



class ProcessingStep(ABC):
    def __init__(self, next_step=None):
        self.next_step = next_step

    def set_next(self, next_step):
        self.next_step = next_step
        return next_step  # chaining helper

    def process(self, rgb_frame, depth_frame):
        rgb_out, depth_out = self._process(rgb_frame, depth_frame)
        if self.next_step:
            return self.next_step.process(rgb_out, depth_out)
        else:
            return rgb_out, depth_out

    @abstractmethod
    def _process(self, rgb_frame, depth_frame):
        pass


class DepthClampAndMask(ProcessingStep):
    """
    Clamp depth to a valid metric range and set everything else to 0 (invalid).
    Incoming depth assumed uint16 in millimeters (0 = invalid). If float (m), convert.
    Constructor expects meters.
    """
    def __init__(self, z_min_m=0.25, z_max_m=3.5):
        super().__init__()
        self.min_mm = int(max(0, round(z_min_m * 1000.0)))
        self.max_mm = int(max(0, round(z_max_m * 1000.0)))

    def _process(self, rgb_frame, depth_frame):
        if depth_frame is None:
            return rgb_frame, depth_frame

        if np.issubdtype(depth_frame.dtype, np.floating):
            depth_mm = (np.clip(depth_frame, 0.0, 65.535) * 1000.0).astype(np.uint16)
        else:
            depth_mm = depth_frame

        valid = depth_mm > 0
        out = depth_mm.copy()
        out[valid & (out < self.min_mm)] = 0
        out[valid & (out > self.max_mm)] = 0
        return rgb_frame, out


class LocalMedianReject(ProcessingStep):
    """
    Suppress isolated depth spikes by comparing to a local median.
    win: odd kernel (3,5,7), thr_mm: reject if |d - median| > thr.
    """
    def __init__(self, win=5, thr_mm=60):
        super().__init__()
        self.win = int(win if win % 2 == 1 else win + 1)
        self.thr = int(thr_mm)

    def _process(self, rgb, depth_u16):
        if depth_u16 is None:
            return rgb, depth_u16
        med = cv2.medianBlur(depth_u16, self.win)
        diff = cv2.absdiff(depth_u16, med)
        out = depth_u16.copy()
        valid = depth_u16 > 0
        out[valid & (diff > self.thr)] = 0
        return rgb, out


class CropROI(ProcessingStep):
    """
    Crop a rectangular ROI (x0,y0,w,h) on RGB and depth.
    NOTE: this step does NOT know intrinsics; Streamer.py will adjust cx,cy.
    """
    def __init__(self, x0: int, y0: int, w: int, h: int):
        super().__init__()
        self.x0 = int(max(0, x0))
        self.y0 = int(max(0, y0))
        self.w  = int(max(1, w))
        self.h  = int(max(1, h))

    def _process(self, rgb, depth):
        if depth is not None:
            H, W = depth.shape[:2]
        elif rgb is not None:
            H, W = rgb.shape[:2]
        else:
            return rgb, depth

        x1 = min(self.x0 + self.w, W)
        y1 = min(self.y0 + self.h, H)
        x0 = min(self.x0, x1 - 1)
        y0 = min(self.y0, y1 - 1)

        if rgb is not None:
            rgb = rgb[y0:y1, x0:x1].copy()
        if depth is not None:
            depth = depth[y0:y1, x0:x1].copy()
        return rgb, depth


class EncodeRGBAsJPEG(ProcessingStep):
    """Optional: turn RGB into a JPEG buffer (uint8)."""
    def _process(self, rgb_frame, depth_frame):
        if rgb_frame is None:
            return None, depth_frame
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        ret, rgb_buf = cv2.imencode('.jpg', rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return rgb_buf, depth_frame


class DownSampling(ProcessingStep):
    """
    Block downsample using skimage.block_reduce.
    mode='avg' for RGB, 'min' for depth (keeps close structure).
    NOTE: Streamer.py will scale fx,fy,cx,cy; this step only resizes arrays.
    """
    def __init__(self, blocksize):
        super().__init__()
        self.block_size = int(max(1, blocksize))

    def _process(self, rgb_frame, depth_frame):
        depth = self.downsample(depth_frame, mode='min') if depth_frame is not None else None
        rgb   = self.downsample(rgb_frame,  mode='avg') if rgb_frame is not None else None
        return rgb, depth

    def downsample(self, img, mode='avg'):
        func = np.mean if mode == 'avg' else np.min
        if img.ndim == 3:  # H×W×C
            return block_reduce(img, block_size=(self.block_size, self.block_size, 1), func=func).astype(img.dtype)
        else:
            return block_reduce(img, block_size=(self.block_size, self.block_size), func=func).astype(img.dtype)



Clamp   = DepthClampAndMask
Median  = LocalMedianReject
ROICrop = CropROI

class Downsample(DownSampling):
    def __init__(self, stride=2):
        super().__init__(blocksize=int(stride))



def build_default_steps(global_cfg: dict):
    """
    Clamp → (Median) → (ROI) → (Downsample)
    YAML keys:
      depth_min_mm, depth_max_mm, median_ksize, roi_xywh: [x,y,w,h], downsample_stride
    """
    steps = []
    dmin_mm = global_cfg.get("depth_min_mm")
    dmax_mm = global_cfg.get("depth_max_mm")
    if dmin_mm is not None or dmax_mm is not None:
        zmin_m = (dmin_mm or 0) / 1000.0
        zmax_m = (dmax_mm or 65535) / 1000.0
        steps.append(Clamp(z_min_m=zmin_m, z_max_m=zmax_m))

    k = int(global_cfg.get("median_ksize", 0) or 0)
    if k > 0:
        steps.append(Median(win=k, thr_mm=60))

    if "roi_xywh" in global_cfg and global_cfg["roi_xywh"]:
        x, y, w, h = map(int, global_cfg["roi_xywh"])
        steps.append(ROICrop(x, y, w, h))

    s = int(global_cfg.get("downsample_stride", 1) or 1)
    if s > 1:
        steps.append(Downsample(stride=s))

    return steps
