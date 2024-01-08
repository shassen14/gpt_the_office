import utils
import config.base as cfg
import gpt.singlehead as sh
import gpt.multihead as mh

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_printoptions(precision=2, linewidth=100)

batch = 4
time = 8
channels = 32

head_size = 16

if __name__ == "__main__":
    # testing code stuff
    # input, next = utils.get_batch("shakespeare",
    #                               "train.bin",
    #                               cfg.block_size,
    #                               cfg.batch_size,
    #                               cfg.device_type)

    sh.Attention(cfg.block_size, cfg.n_embeddings, cfg.head_size, cfg.dropout)

    mh.Attention(cfg.block_size, cfg.n_embeddings, cfg.head_size, cfg.num_heads, cfg.dropout)


    # x = torch.randn(batch, time, channels)  # B, T, C
    
    # # single head
    # key = nn.Linear(channels, head_size, bias=False)    # C -> H
    # query = nn.Linear(channels, head_size, bias=False)  # C -> H

    # k = key(x)      # B, T, H
    # q = query(x)    # B, T, H

    # weights = q @ k.transpose(1, 2) # (B, T, H) @ (B, H, T) = (B, T, T)

    # # lower triangle of (T, T) matrix with 1s
    # # then replace the upper triangle of 0s in tril to -inf for weights
    # tril = torch.tril(torch.ones(time, time))
    # weights = weights.masked_fill(tril == 0, float('-inf'))
    
    # # weights = F.softmax(weights, dim=-1)
    # output = weights @ x    # (B, T, T) @ (B, T, C) = (B, T, C) 

    # print(x.shape)
    # print(k.shape)
    # # print(weights[0])
    # print(output[0])


    print("Hello World!")