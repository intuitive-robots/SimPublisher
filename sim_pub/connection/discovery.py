from time import sleep
import threading
from threading import Thread
from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
from typing import Optional


class DiscoveryThread:
  
  def __init__(self, message : str, port : int , intervall=2):
    self.port = port
    self.running = True
    self.intervall = intervall
    self.message = message.encode()
    self.thread = Thread(target=self._loop)
    self.socket = socket(AF_INET, SOCK_DGRAM) #create UDP socket
    self.socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1) #this is a broadcast socket

  def _loop(self):
    print("* Discovery is running on port", self.port)
    while self.running:
      self.socket.sendto(self.message, ("255.255.255.255", self.port))
      sleep(self.intervall)
  
  def start(self):
    self.thread.start()

  def stop(self):
    self.running = False
    self.thread.join()