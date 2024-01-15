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

# @torch.no_grad()
# def estimate_loss(model,
#                   ):
#     out = {}
#     model.eval()
#     for split in ['train.bin', 'val.bin']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

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

    # create model using config params
    # convert model to the device. important if using cuda
    model = gm.GPT(meta_vocab_size,
                   cfg.num_layers,
                   cfg.block_size,
                   cfg.n_embeddings,
                   cfg.head_size,
                   cfg.num_heads,
                   cfg.dropout,
                   cfg.device_type)
    m = model.to(cfg.device_type)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'million parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Iterate max iteration amount of times 
    for i in range(cfg.max_iterations):
        xb, yb = utils.get_batch('shakespeare',
                                 'train.bin',
                                 cfg.block_size,
                                 cfg.batch_size,
                                 cfg.device_type)
        
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % cfg.eval_iterations == 0 or i == cfg.max_iterations - 1:
            print(loss)
