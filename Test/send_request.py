import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:7721")

socket.send_string("SaveRecord")
message = socket.recv_string()
print(message)


