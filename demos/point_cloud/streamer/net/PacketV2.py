import struct
import numpy as np

MAGIC = 0xABCD1234
VERSION = 2
FLAG_POSE = 1      
FLAG_INTR = 2      

class PacketV2Writer:
    """
    Packs a message:
    [Header 36B]
    [Intrinsics 16B if flags&2]
    [RGB JPEG rgb_len B]
    [Depth U16 depth_len B]
    [Pose 4x4 64B if flags&1]
    """
    def __init__(self, send_intrinsics: bool = False):
        self.send_intrinsics = send_intrinsics

    def pack(
        self,
        camera_id: int,
        timestamp_us: int,
        width: int,
        height: int,
        rgb_jpeg_bytes: bytes,
        depth_u16: np.ndarray,
        intrinsics: np.ndarray | None,
        pose_Twc: np.ndarray | None
    ) -> bytes:
        flags = 0
        if pose_Twc is not None:
            flags |= FLAG_POSE
        if self.send_intrinsics and intrinsics is not None:
            flags |= FLAG_INTR

        depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
        depth_bytes = depth_u16.tobytes(order="C")

        header = struct.pack(
            "<IHHI Q I I I I",
            MAGIC, VERSION, flags, int(camera_id),
            int(timestamp_us), int(width), int(height),
            len(rgb_jpeg_bytes), len(depth_bytes)
        )

        parts = [header]

        if flags & FLAG_INTR:
            intr = np.asarray(intrinsics, dtype=np.float32).reshape(4)
            parts.append(struct.pack("<4f", *intr))

        parts.append(rgb_jpeg_bytes)
        parts.append(depth_bytes)

        if flags & FLAG_POSE:
            T = np.asarray(pose_Twc, dtype=np.float32).reshape(16)
            parts.append(struct.pack("<16f", *T))

        return b"".join(parts)
