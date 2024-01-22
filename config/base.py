"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
import torch

choose_config = 'small_cfg'

if choose_config == 'test_cfg':
    import config.test_cfg as cfg
elif choose_config == 'small_cfg':
    import config.small_cfg as cfg

# Dataset to utilize
dataset_dir     = cfg.dataset_dir
train_file      = cfg.train_file
val_file        = cfg.val_file
pkl_file        = cfg.pkl_file
file_array      = [cfg.train_file, val_file]

# Parameter Save/Load
param_dir       = cfg.param_dir
pt_file         = cfg.pt_file

## Generate sample texts
sample_dir      = cfg.sample_dir
sample_file     = cfg.sample_file
max_new_tokens  = cfg.max_new_tokens

############################## Paramaters #####################################
block_size      = cfg.block_size
batch_size      = cfg.batch_size
num_embeddings  = cfg.num_embeddings
num_heads       = cfg.num_heads
num_layers      = cfg.num_layers
head_size       = num_embeddings // num_heads
dropout         = cfg.dropout

learning_rate   = cfg.learning_rate
max_iterations  = cfg.max_iterations
eval_iterations = int(max_iterations / 10)

# get device type. get GPU or apple if possible
device_type = 'cpu'

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = 'mps'

print("Device Type: " + device_type)