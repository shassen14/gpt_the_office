"""
Choose configuration to use
"""
import torch

# get device type
device_type = 'cpu'

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = 'mps'

choose_config = 'cfg1'

if choose_config == 'cfg1':
    import config.cfg1 as cfg

batch_size = cfg.batch_size
block_size = cfg.block_size