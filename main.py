import utils
import config.base as cfg

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

if __name__ == "__main__":
    # testing code stuff
    utils.get_batch("the_office",
                    "train.bin",
                    cfg.batch_size,
                    cfg.block_size,
                    "cuda")

    print("Hello World!")