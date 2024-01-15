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

# Dataset to utilize
dataset_dir     = cfg.dataset_dir
train_file      = cfg.train_file
val_file        = cfg.val_file
pkl_file        = cfg.pkl_file
file_array      = [cfg.train_file, val_file]

############################## Paramaters #####################################
block_size      = cfg.block_size
batch_size      = cfg.batch_size
n_embeddings    = cfg.n_embeddings

num_heads       = cfg.num_heads
head_size       = n_embeddings // num_heads


dropout         = cfg.dropout

num_layers      = cfg.num_layers

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