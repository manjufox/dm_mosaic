import sys
sys.path.append("..")
sys.path.append(".")

import pprint
from config import TrainingConfig

config = TrainingConfig()

def dprint(*args):
    if config.debug:
        pprint.pprint(args)


