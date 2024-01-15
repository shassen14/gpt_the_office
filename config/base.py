"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
import torch

choose_config = 'test_cfg'

if choose_config == 'test_cfg':
    import config.test_cfg as cfg

# TODO: organize and comment

############################## Paramaters #####################################
block_size      = cfg.block_size
batch_size      = cfg.batch_size
n_embeddings    = cfg.n_embeddings

head_size       = cfg.head_size
num_heads       = cfg.num_heads

dropout         = cfg.dropout

num_layers      = cfg.num_layers


# get device type. get GPU or apple if possible
device_type = 'cpu'

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = 'mps'

print("Device Type: " + device_type)