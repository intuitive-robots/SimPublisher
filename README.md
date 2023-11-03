# SimPublisher

A cross-environment tool to publish objects from simulation for Augmented Reality and Human Robot Interaction.

## Introduction

This repo uses websockets to create a server and publishers for publishing object states from simulation,
and it could also receive and process message from other application by websockets.
For instance, Unity renders GameObjects in real time from the stream and sends the interaction message to simulation. 

## Installation

Install the dependency
```
pip install websockets
```

## Usage

Primitive module is used as a basic implementation of server and streamer.

```py
server = PrimitiveServer()
server.start_server_thread(block=True)
```
