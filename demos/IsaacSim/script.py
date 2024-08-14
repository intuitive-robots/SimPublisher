# these code should only be run with the isaac sim vscode extension

import omni
import omni.usd

from pxr import Usd, UsdUtils, UsdShade
from usdrt import Usd as RtUsd
from usdrt import UsdGeom as RtGeom
from usdrt import Rt

stage = omni.usd.get_context().get_stage()
print(stage)

stage_id = UsdUtils.StageCache.Get().Insert(stage)
stage_id = stage_id.ToLongInt()
print(stage_id)

rtstage = RtUsd.Stage.Attach(stage_id)
print(rtstage)

prim_path = "/World/defaultGroundPlane/Environment"
prim = stage.GetPrimAtPath(prim_path)
print(prim)
print(prim.GetTypeName())

matapi: UsdShade.MaterialBindingAPI = UsdShade.MaterialBindingAPI(prim)
print(matapi)
mat: UsdShade.Material = matapi.GetDirectBinding().GetMaterial()
print(mat)
mat_prim: Usd.Prim = stage.GetPrimAtPath(mat.GetPath())
print(mat_prim)
