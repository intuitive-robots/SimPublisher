from sim_pub.server.ws.ws_server import *

primitive_server = PrimitiveServer()
primitive_server.start_server_thread(block=True)

