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

prim_path = "/Cube"
prim = stage.GetPrimAtPath(prim_path)
print(prim)
print(prim.GetTypeName())

matapi: UsdShade.MaterialBindingAPI = UsdShade.MaterialBindingAPI(prim)
print(matapi)
mat: UsdShade.Material = matapi.GetDirectBinding().GetMaterial()
print(mat)
mat_prim: Usd.Prim = stage.GetPrimAtPath(mat.GetPath())
print(mat_prim)

uvs = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), dtype=np.float32)
print(uvs)
print(len(uvs))

for i in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
    print(i.GetName())
