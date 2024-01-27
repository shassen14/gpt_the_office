# TODO: organize and comment

# dataset
dataset_dir = "data/shakespeare_tiktoken"
train_file = "train.bin"
val_file = "val.bin"
pkl_file = "meta.pkl"

# params
param_dir = "params"
pt_file = "test_tiktoken_cfg.pt"

# generate sample
sample_dir = "examples"
sample_file = "shakespeare_tiktoken_example.txt"
max_new_tokens = 2000

# Model
batch_size = 4
block_size = 512
num_embeddings = 768
num_heads = 8
num_layers = 8
dropout = 0.2

# Learning
learning_rate = 3e-4
max_iterations = 1000
