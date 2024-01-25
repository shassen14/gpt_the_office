# TODO: organize and comment

# dataset
dataset_dir = 'data/shakespeare'
train_file = 'train.bin'
val_file = 'val.bin'
pkl_file = 'meta.pkl'

# params
param_dir = 'params'
pt_file = 'small_cfg.pt'

# generate sample
sample_dir = 'examples'
sample_file = 'shakespeare_small_cfg_example.txt'
max_new_tokens = 1000

# Model
batch_size = 4
block_size = 8
num_embeddings = 32
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning
learning_rate = 3e-3
max_iterations = 1000
