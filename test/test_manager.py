from simpub.core.simpub_manager import SimPubManager
from simpub.core.subscriber import Subscriber
from simpub.core.log import logger
a = SimPubManager()
import time
time.sleep(1)
s = Subscriber("UnityLog", lambda x: print(x))
a.join()
