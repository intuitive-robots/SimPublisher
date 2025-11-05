# SimPublisher

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

A versatile Python tool designed to seamlessly render objects from simulations or sensors to Mixed Reality (MR) and Augmented Reality (AR) headsets.

## Table of Contents
- [Introduction](#Introduction)
- [Features](#Features)
- [Installation](#Installation)
- [Usage](#Usage)

## Introduction

Are you looking to integrate your Human-Robot Interaction (HRI) application with MR/VR/AR headsets effortlessly?
This repository is perfect for you.

We provide a ROS-style interface that allows you to easily use Python code to project simulation scenes onto MR headsets. Additionally, you can use input data from the headset to control virtual robots and objects within the simulation environment or even real robots.

This repository uses ZMQ to communicate with MR applications, featuring automatic device discovery and reconnecting capabilities.

Currently we support two hardware platforms including [Meta Quest 3](https://github.com/intuitive-robots/IRIS-Meta-Quest3) and HoloLens 2.
These applications are built based on a Unity Package named [IRIS-Viz](https://github.com/intuitive-robots/IRIS-Viz), which is a Unity Package that is easy to deploy for projecting and updating simulation scenes in MR headsets.

## Features

### Easy Connection to Headset

- **Automatic Device Discovery**: SimPub will search for all devices in the subnet and connect to them fully automatically.
- **Auto-Reconnecting**: Automatically reconnects to the PC if you shutdown the Python script.
- **Remote Logger**: The log will be sent to the PC simulation, including FPS and latency information.

### Supported Simulation Environments

- [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html)
- [SimulationFramework](https://github.com/ALRhub/SimulationFrameworkPublic)
- [FancyGym](https://github.com/ALRhub/fancy_gym)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)

### Supported Headsets

- [Meta Quest 3](https://github.com/intuitive-robots/IRIS-Meta-Quest3)
- HoloLens 2

## Installation

1. Install the dependencies:
```bash
pip install zmq trimesh
```

2. Install this repository:
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
We recommend using a single WiFi router for PC-headset communication to ensure optimal connectivity.
Additionally, using a wired cable to connect your PC can significantly reduce latency.

3. Run the usage examples under the folder `/demos/`, then wear the headset, start the Unity application and enjoy!

### Isaac Sim
Please check [this](./demos/IsaacSim/README.md) for information on how to set up SimPub with Isaac Sim.
