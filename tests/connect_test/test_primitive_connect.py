from sim_pub.server.ws.WsServer import *

primitive_server = PrimitiveServer()
primitive_server.start_server_thread(block=True)

