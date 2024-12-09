from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Mesh:
    vertex_buf: npt.NDArray = np.array([])
    normal_buf: npt.NDArray = np.array([])
    uv_buf: npt.NDArray = np.array([])
    index_buf: npt.NDArray = np.array([])


def split_mesh_by_uv_islands(input_mesh: Mesh) -> Mesh:
    assert type(input_mesh.vertex_buf) is np.ndarray
    assert type(input_mesh.normal_buf) is np.ndarray
    assert type(input_mesh.uv_buf) is np.ndarray
    assert type(input_mesh.index_buf) is np.ndarray

    assert input_mesh.vertex_buf.shape[1] == 3
    assert input_mesh.normal_buf.shape[1] == 3
    assert input_mesh.uv_buf.shape[1] == 2
    assert input_mesh.index_buf.shape[1] in {3, 4}
