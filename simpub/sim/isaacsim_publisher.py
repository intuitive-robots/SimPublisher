import io

import numpy as np
from pxr import Usd
from usdrt import Rt
from usdrt import UsdGeom as RtGeom

from ..core.net_component import ByteStreamer
from ..core.simpub_server import SimPublisher
from ..parser.isaacsim import IsaacSimStageParser


class IsaacSimPublisher(SimPublisher):
    def __init__(
        self,
        host: str,
        stage: Usd.Stage,
        ignored_prim_paths: list[str] = [],
        texture_cache_dir: str = None,
    ) -> None:
        self.parser = IsaacSimStageParser(
            stage=stage,
            ignored_prim_paths=ignored_prim_paths,
            texture_cache_dir=texture_cache_dir,
        )
        self.sim_scene = self.parser.parse_scene()

        # set usdrt stage and tracked prims
        self.rt_stage = self.parser.get_usdrt_stage()
        self.tracked_prims, self.tracked_deform_prims = self.parser.get_tracked_prims()

        self.sim_scene.process_sim_obj(self.sim_scene.root)

        super().__init__(self.sim_scene, host)

        # add deformable update streamer
        self.deform_update_streamer = ByteStreamer("DeformUpdate", self.get_deform_update, start_streaming=True)

    def get_update(self) -> dict[str, list[float]]:
        state = {}
        for tracked_prim in self.tracked_prims:
            prim_name = tracked_prim["name"]
            prim_path = tracked_prim["prim_path"]
            # get prim with usdrt api (necessary for getting updates from physics simulation)
            rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
            rt_prim = Rt.Xformable(rt_prim)
            pos = rt_prim.GetWorldPositionAttr().Get()
            rot = rt_prim.GetWorldOrientationAttr().Get()
            # convert pos and rot to unity coord system
            state[prim_name] = [
                pos[1],
                pos[2],
                -pos[0],
                -rot.GetImaginary()[1],
                -rot.GetImaginary()[2],
                rot.GetImaginary()[0],
                rot.GetReal(),
            ]
        return state

    def get_deform_update(self) -> bytes:
        ###################################################
        ## Write binary data to send as update
        ## Structure:
        ##
        ##   L: Length of update string containing all deformable meshes [ 4 bytes ]
        ##   S: Update string, semicolon seperated list of meshes contained in this update [ ? bytes ]
        ##   N: Number of verticies for each mesh [ num_meshes x 4 bytes]
        ##   V: Verticies for each mesh [ ? bytes for each mesh ]
        ##
        ##       | L | S ... S | N ... N | V ... V |
        #####################################################

        state = {}
        mesh_vert_len = np.ndarray(len(self.tracked_deform_prims), dtype=np.uint32)  # N
        update_list = ""  # S

        for idx, tracked_prim in enumerate(self.tracked_deform_prims):
            mesh_name = tracked_prim["name"]
            prim_path = tracked_prim["prim_path"]

            vertices = np.asarray(
                self.rt_stage.GetPrimAtPath(prim_path).GetAttribute(RtGeom.Tokens.points).Get(),
                dtype=np.float32,
            )
            vertices = vertices[:, [1, 2, 0]]
            vertices[:, 2] = -vertices[:, 2]
            vertices = vertices.flatten()
            state[mesh_name] = np.ascontiguousarray(vertices)

            mesh_vert_len[idx] = vertices.size
            update_list += f"{mesh_name};"

        update_list = str.encode(update_list)
        bin_buffer = io.BytesIO()

        bin_buffer.write(len(update_list).to_bytes(4, "little"))  # L
        bin_buffer.write(update_list)  # S

        bin_buffer.write(mesh_vert_len)  # N

        for hash in state:
            bin_buffer.write(state[hash])  # V

        return bin_buffer.getvalue()
