# SimPublisher

A cross-environment tool to publish objects from simulation for Augmented Reality and Human Robot Interaction.

## Introduction

This repo uses zmq to create a server and publishers for publishing object states from simulation,
and it could also receive and process message from other application.
For instance, Unity renders GameObjects in real time from the stream and sends the interaction message to simulation, and please find this Unity project [here](https://github.com/intuitive-robots/IRXR-Unity).

## Installation

Install the dependency
```
pip install zmq trimesh
```

Install this repo
```
cd [the path to this project]
pip install -e .
```

## Usage

This repo provides some open-and-use tools for simulation environment including:
- [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html),
- [SimulationFramework](https://github.com/ALRhub/SimulationFrameworkPublic),

We are working on these simulation environment and it will be coming soon!
- [FancyGym](https://github.com/ALRhub/fancy_gym)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)

Please find the usage example under the folder `/demos/`.
