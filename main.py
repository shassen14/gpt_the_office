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



if __name__ == "__main__":
    # obtain file paths
    pickle_path = utils.get_file_path(cfg.dataset_dir,
                                      cfg.pkl_file)
    pt_path = utils.get_file_path(cfg.param_dir, cfg.pt_file)

    # obtain vocabulary size from pkl
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        meta_encode = lambda s: [meta['stoi'][c] for c in s] # encoder: take a string, output a list of integers
        meta_decode = lambda l: ''.join([meta['itos'][i] for i in l]) # decoder: take a list of integers, output a string
        print(f"found vocab_size = {meta_vocab_size} (inside {pickle_path})")
    else:
        print(pickle_path + " doesn't exist. Please give a valid one.")
        exit()

    # create model using config params
    # convert model to the device. important if using cuda
    model = gm.GPT(meta_vocab_size,
                   cfg.num_layers,
                   cfg.block_size,
                   cfg.num_embeddings,
                   cfg.head_size,
                   cfg.num_heads,
                   cfg.dropout,
                   cfg.device_type)
    model.to(cfg.device_type)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'million parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # iterate max iteration amount of times 
    for i in range(cfg.max_iterations):

        if i % cfg.eval_iterations == 0 or i == cfg.max_iterations - 1:
            losses = utils.estimate_loss(model, cfg)
            print(f"step {i}: train loss {losses[cfg.train_file]:.4f}, val loss {losses[cfg.val_file]:.4f}")

        xb, yb = utils.get_batch(cfg.dataset_dir,
                                 cfg.train_file,
                                 cfg.block_size,
                                 cfg.batch_size,
                                 cfg.device_type)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # save model
        torch_model = {
            'iteration': i,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'config': cfg,
            # 'model_args': model_args,
            # 'best_val_loss': best_val_loss,
        }
        torch.save(torch_model, pt_path)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device_type)
    print(meta_decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # open('example.txt', 'w').write(meta_decode(m.generate(context, max_new_tokens=30000)[0].tolist()))