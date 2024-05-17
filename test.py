
import math
import numpy as np
from simpub.serialize import serialize_data
import zmq
from simpub import SimPublisher, SimScene, mj2pos
import mujoco as mj

from simpub.udata import UJointType, UScene

SERVICE_PORT = 5521 # this port is configured in the firewall 
DISCOVERY_PORT = 5520
STREAMING_PORT = 5522

scene = SimScene.from_file("models copy/mujoco/surroundings/kit_lab_surrounding.xml")

publisher = SimPublisher(scene, service_port=SERVICE_PORT, streaming_port=STREAMING_PORT, discovery_port=DISCOVERY_PORT)


publisher.start()


while True:
  ...