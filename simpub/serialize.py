
import json
from typing import Any

import numpy as np
import dataclasses as dc

class _CustomEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif dc.is_dataclass(obj):
      result : dict = dc.asdict(obj)
      hidden = [key for key in result if key.startswith("_")]
      for key in hidden:
        del result[key]
      return result
    else:
      return super().default(obj)


def serialize_data(data : Any, **jsonkwargs) -> str:
  if isinstance(data, str): return data
  return json.dumps(data, separators=(',', ':'), cls=_CustomEncoder, **jsonkwargs)