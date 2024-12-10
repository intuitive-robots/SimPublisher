from simpub.core import init_net_manager

manager = init_net_manager(host="192.168.0.134")

manager.start()
manager.spin()

