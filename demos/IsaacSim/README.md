# Isaac Lab Setup Guide

## Quick Start

Please follow the [readme](../../README.md) in the root folder for installing simpub.

You can install Isaac Sim and Isaac Lab following [this page](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html). You could also install isaac sim with pip, but I didn't test on it.

Now you can activate your Isaac Lab conda environment, and run some demos. For example, run this in the root folder:
```
python demos/IsaacSim/spawn_demo.py
```

- [spawn_demo.py](./spawn_demo.py) is an example of spawing primitives manually.
- [env_demo.py](./env_demo.py) is an example of using environments.
- [meta_quest_3_ik_demo.py](./meta_quest_3_ik_demo.py) is an interactive scene where you can control an robot arm with joysticks with meta quest 3. (It might not work properly...)

To use simpub with Isaac Sim in your code, simply add this after the scene has finished initialization (please check the demos):
```
from simpub.sim.isaacsim_publisher import IsaacSimPublisher

# use only one of these
publisher = IsaacSimPublisher(host="192.168.0.134", stage=sim.stage) # for InteractiveScene
publisher = IsaacSimPublisher(host="192.168.0.134", stage=env.sim.stage) # for environments
```

## Tips

The file [demos/IsaacSim/script.py](./script.py) provides some information on how to retrieve materials from usd primitives. But it's not complete yet. You still have to figure out how to retireve albedo color and texture from the Material object. You might find [this](https://openusd.org/dev/api/class_usd_shade_material_binding_a_p_i.html) helpful (or not).

Currently simpub doesn't support normal indexing. So one vertex will only have one normal, which is the cause of the rendering artifacts. Per-vertex normals are smoothed, so you can't render flat surfaces and sharp edges. To solve this, triangle faces in meshes should not only index vertices, but also normals. This means a vertex will have different normals on different triangles.

Check [this](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html) for the different between usd, usdrt and fabric api. Basically, in python, you should use usd for cpu simulation and usdrt for gpu simulation.

In omniverse, there are primitive prototypes that can be instantiated. Currently, IsaacSimPublisher doesn't handle this properly. It'll create a different mesh object and store the same geometry repeatedly for all instances of a prototype. For large scenes with complex geometries, this might be a problem.

Currently usdrt stage are created for both cpu and gpu simulation. This works but might result in unexpected things. Maybe add a new parameter to the publisher indicating where the simulation is running and use only usd stage for cpu simulation.
