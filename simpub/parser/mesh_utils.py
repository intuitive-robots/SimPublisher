from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Mesh:
    vertex_buf: npt.NDArray = np.array([])
    normal_buf: npt.NDArray = np.array([])
    index_buf: npt.NDArray = np.array([])
    uv_buf: npt.NDArray | None = None

    def __post_init__(self):
        if not self.uv_buf:
            self.uv_buf = None

        assert type(self.vertex_buf) in {np.ndarray, list}
        assert type(self.normal_buf) in {np.ndarray, list}
        assert type(self.index_buf) in {np.ndarray, list}
        assert self.uv_buf is None or type(self.uv_buf) in {np.ndarray, list}

        self.vertex_buf = np.array(self.vertex_buf)
        self.normal_buf = np.array(self.normal_buf)
        self.index_buf = np.array(self.index_buf)
        if self.uv_buf is not None:
            self.uv_buf = np.array(self.uv_buf)

        assert self.vertex_buf.shape[1] == 3
        assert self.normal_buf.shape[1] == 3
        assert self.index_buf.shape[1] in {3, 4}
        if self.uv_buf is not None:
            assert self.uv_buf.shape[1] == 2

        # for isaac sim format only...
        assert self.vertex_buf.shape[0] <= self.normal_buf.shape[0]
        if self.uv_buf is not None:
            assert self.normal_buf.shape[0] == self.uv_buf.shape[0]


def split_mesh_faces(input_mesh: Mesh) -> Mesh:
    # no need to process if each vertex has only one normal
    if input_mesh.vertex_buf.shape[0] == input_mesh.normal_buf.shape[0]:
        return input_mesh

    vertex_buf = []
    normal_buf = []
    uv_buf = []
    index_buf = []

    for tri in input_mesh.index_buf:
        if len(tri) == 3:
            index_buf.append(
                [
                    len(vertex_buf) + 0,
                    len(vertex_buf) + 1,
                    len(vertex_buf) + 2,
                ]
            )
        else:
            assert len(index_buf) == 4
            index_buf.append(
                [
                    len(vertex_buf) + 0,
                    len(vertex_buf) + 1,
                    len(vertex_buf) + 2,
                    len(vertex_buf) + 3,
                ]
            )

        vertex_buf.append(input_mesh.vertex_buf[tri[0]])
        vertex_buf.append(input_mesh.vertex_buf[tri[1]])
        vertex_buf.append(input_mesh.vertex_buf[tri[2]])
        if len(tri) == 4:
            vertex_buf.append(input_mesh.vertex_buf[tri[3]])

        normal_buf.append(input_mesh.normal_buf[tri[0]])
        normal_buf.append(input_mesh.normal_buf[tri[1]])
        normal_buf.append(input_mesh.normal_buf[tri[2]])
        if len(tri) == 4:
            normal_buf.append(input_mesh.normal_buf[tri[3]])

        if input_mesh.uv_buf is not None:
            uv_buf.append(input_mesh.uv_buf[tri[0]])
            uv_buf.append(input_mesh.uv_buf[tri[1]])
            uv_buf.append(input_mesh.uv_buf[tri[2]])
            if len(tri) == 4:
                uv_buf.append(input_mesh.uv_buf[tri[3]])

    return Mesh(
        vertex_buf=vertex_buf,
        normal_buf=normal_buf,
        index_buf=index_buf,
        uv_buf=uv_buf or None,
    )
