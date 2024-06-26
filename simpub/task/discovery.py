from time import sleep
from threading import Thread
from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_REUSEADDR

BROADCAST_MASK = "255.255.255.255"


class DiscoverTask:
    
    def __init__(self, callback, port: int, message: str, intervall=2):
        self._port = port
        self._running = True
        self._intervall = intervall
        self._message = message.encode()
        self._callback = callback
        self._thread = Thread(target=self._loop)


class DiscoveryReceiver:

    def __init__(self, callback, port: int):
        self._port = port
        self._running = False
        self._callback = callback
        self._thread = Thread(target=self._loop)

    def _loop(self):
        # create UDP socket
        self.conn = socket(AF_INET, SOCK_DGRAM)
        self.conn.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.conn.bind(('', self._port))
        self._running = True
        while self._running:
            message, server = self.conn.recvfrom(1028)
            self._callback(message.decode(), server[0])

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False
        self.conn.close()
        self._thread.join()
