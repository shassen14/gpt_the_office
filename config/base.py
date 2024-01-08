"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
import torch

choose_config = 'cfg1'

if choose_config == 'cfg1':
    import config.cfg1 as cfg

############################## Paramaters #####################################
block_size      = cfg.block_size
batch_size      = cfg.batch_size
n_embeddings    = cfg.n_embeddings

head_size       = cfg.head_size
num_heads       = cfg.num_heads

dropout         = cfg.dropout


# get device type. get GPU or apple if possible
device_type = 'cpu'

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = 'mps'

print("Device Type: " + device_type)