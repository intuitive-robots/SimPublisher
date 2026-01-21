import time, threading
from typing import Optional, Tuple
import numpy as np
import cv2
import zmq

from net.PacketV2 import PacketV2Writer

# ====================== Preview (RGB + Depth) ======================
def _resolve_colormap_code(name_or_code):
    if isinstance(name_or_code, int):
        return int(name_or_code)
    name = str(name_or_code).strip().upper()
    return getattr(cv2, f"COLORMAP_{name}", cv2.COLORMAP_JET)

def _colorize_depth_mm(depth_u16: np.ndarray, dmin: int, dmax: int, cmap_code: int) -> np.ndarray:
    d = np.asarray(depth_u16, dtype=np.float32)
    valid = d > 0
    lo, hi = float(dmin), float(max(dmax, dmin + 1))
    d = np.clip(d, lo, hi)
    norm = (d - lo) * (255.0 / (hi - lo))
    norm[~valid] = 0.0
    img8 = norm.astype(np.uint8)
    cm = cv2.applyColorMap(img8, cmap_code)
    cm[~valid] = (0, 0, 0)
    return cm

class PreviewHub(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled_rgb = False
        self.enabled_depth = False
        self._run = False
        self._lock = threading.Lock()
        self._rgb = {}
        self._depth = {}

    def enable(self, rgb: bool, depth: bool):
        self.enabled_rgb = bool(rgb)
        self.enabled_depth = bool(depth)
        if (self.enabled_rgb or self.enabled_depth) and not self._run:
            self._run = True
            self.start()

    def stop(self):
        self._run = False


    def update(self, cam_id: int, bgr: np.ndarray | None, depth_bgr: np.ndarray | None):
        if not self._run:
            return
        with self._lock:
            if self.enabled_rgb and bgr is not None:
                self._rgb[cam_id] = bgr
            if self.enabled_depth and depth_bgr is not None:
                self._depth[cam_id] = depth_bgr

    def run(self):
        while self._run:
            imgs = []
            with self._lock:
                if self.enabled_rgb:
                    imgs.extend(("RGB", cid, img) for cid, img in self._rgb.items())
                if self.enabled_depth:
                    imgs.extend(("Depth", cid, img) for cid, img in self._depth.items())

            for kind, cid, img in imgs:
                cv2.imshow(f"{kind} - Cam {cid}", img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC aborts preview loop
                self._run = False
                break

        cv2.destroyAllWindows()


PREVIEW = PreviewHub()

