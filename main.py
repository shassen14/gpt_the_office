import utils
import config.base as cfg
import gpt.model as gm

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os

torch.set_printoptions(precision=2, linewidth=100)

batch = 4
time = 8
channels = 32

head_size = 16

if __name__ == "__main__":
    # obtain vocabulary size from pkl
    pickle_path = utils.get_dataset_path('shakespeare',
                                         'meta.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {pickle_path})")
    else:
        print("pkl file doesn't exist. Please input a valid one.")
        exit()

    model = gm.GPT(meta_vocab_size,
                   cfg.num_layers,
                   cfg.block_size,
                   cfg.n_embeddings,
                   cfg.head_size,
                   cfg.num_heads,
                   cfg.dropout,
                   cfg.device_type)


    # print(chars)

    input, next = utils.get_batch("shakespeare",
                                  "train.bin",
                                  cfg.block_size,
                                  cfg.batch_size,
                                  cfg.device_type)
    
    print(input)