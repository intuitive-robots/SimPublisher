# SimPublisher

This repo is a versatile tool designed to seamlessly rendering objects from simulation or sensors to Mixed Reality (MR) or Augmented Reality (AR) headsets.

## Isaac Sim
Please check [this](./demos/IsaacSim/README.md) for information on how to setup simpub with isaac sim.

## Table of Contents
- [Introduction](#Introduction)
- [Features](#Features)
- [Installation](#Installation)
- [Usage](#Usage)

## Introduction

Are you looking forward to integrate your Human-Robot Interaction (HRI) application with MR/VR/AR headsets effortlessly?
This repository is perfect for you.

We provide a ROS-style interface that allows you to easily use Python code to project simulation scenes onto MR headsets. Additionally, you can use input data from the headset to control virtual robots and objects within the simulation environment or even real robots.

This repo uses zmq to to communicate with MR application with automaticlly searching devices and reconnecting features.

We also offer a [Unity Application](https://github.com/intuitive-robots/IRXR-Unity) that is easy to deploy for projecting and updating simulation scenes in the MR headset.

## Features

### Easy Connection to Headset

- **Automatically Searching**: SimPub will search all the devices in the subnet and connect them fully automatically.
- **Reconnecting**: Reconnecting to the PC if you shutdown the Python script.
- **Remote Logger**: The log will be sent to the PC Simulation including FPS and latency.

### Supported Simulation Environment

- [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html)
- [SimulationFramework](https://github.com/ALRhub/SimulationFrameworkPublic)
- [FancyGym](https://github.com/ALRhub/fancy_gym)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)

### Supported Headset

- Meta Quest3
- HoloLens2

## Installation

1. Install the dependency
```bash
pip install zmq trimesh
```

2. Install this repo
```bash
cd $the_path_to_this_project
pip install -e .
```

## Usage

1. Deploy the Unity application to your headset with the device name.
Please refer to the [website](https://github.com/intuitive-robots/IRXR-Unity).

2. Connect your simulation PC and headset to the same subnet. 
For example, if your simulation PC address is `192.168.0.152`, 
the headset address should share the same prefix, such as `192.168.0.142` or `192.168.0.73`.
We recommend using a single WiFi router for PC-headset communication to ensure optimal connectivity,.
Additionally, using a wired cable to connect your PC can significantly reduce latency.

3. Run the usage examples under the folder `/demos/`, then wear the headset, start the unity application and enjoy!

