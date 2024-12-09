# here you can find some code that may be helpful for retrieving materials from usd primitives
# these code should only be run with the isaac sim vscode extension (maybe not...)

import omni
import omni.usd

from pxr import Usd, UsdUtils, UsdShade, UsdGeom
from usdrt import Usd as RtUsd
from usdrt import UsdGeom as RtGeom
from usdrt import Rt

import numpy as np

stage = omni.usd.get_context().get_stage()
print(stage)

stage_id = UsdUtils.StageCache.Get().Insert(stage)
stage_id = stage_id.ToLongInt()
print(stage_id)

rtstage = RtUsd.Stage.Attach(stage_id)
print(rtstage)

prim_path = "/World/Origin3/Robot/base/visuals/mesh_13"
prim = stage.GetPrimAtPath(prim_path)
print(prim)
print(prim.GetTypeName())

for attr in prim.GetAttributes():
    print(attr)

# uvs = [np.asarray(prim.GetAttribute("primvars:st").Get(), dtype=np.float32)]
# for i in range(1, 100):
#     if not prim.HasAttribute("primvars:st_" + str(i)):
#         break
#     uvs.append(
#         np.asarray(prim.GetAttribute("primvars:st_" + str(i)).Get(), dtype=np.float32)
#     )

# for uv in uvs:
#     print(uv.shape)


matapi: UsdShade.MaterialBindingAPI = UsdShade.MaterialBindingAPI(prim)
print(matapi)

mat: UsdShade.Material = matapi.GetDirectBinding().GetMaterial()
print(mat)

mat_prim: Usd.Prim = stage.GetPrimAtPath(mat.GetPath())
print(mat_prim)

shader = UsdShade.Shader(mat_prim.GetAllChildren()[0])
print(shader)

print(shader.GetInput("diffuse_texture").Get())

# uvs = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), dtype=np.float32)
# print(uvs)
# print(len(uvs))

# for i in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
#     print(i.GetName())
