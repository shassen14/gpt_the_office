import utils
import config.base as cfg

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

if __name__ == "__main__":
    # testing code stuff
    input, next = utils.get_batch("shakespeare",
                                  "train.bin",
                                  cfg.block_size,
                                  cfg.batch_size,
                                  cfg.device_type)

    print("Hello World!")