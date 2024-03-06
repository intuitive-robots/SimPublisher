from sim_pub.model_loader import MJCFLoader

loader = MJCFLoader("./tests/model_test/panda.xml")
[print(k, v) for k, v in loader.default_class_dict.items()]












