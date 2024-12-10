from simpub.core import init_net_manager
from simpub.core.net_component import Publisher

manager = init_net_manager(host="127.0.0.1")
pub_1 = Publisher("test_topic_1")
manager.spin()
