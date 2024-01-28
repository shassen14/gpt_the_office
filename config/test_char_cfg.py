# TODO: organize and comment

# dataset
dataset_dir = "data/shakespeare_char"

# params
param_dir = "params"
pt_file = "test_char_cfg.pt"

# generate sample
sample_dir = "examples"
sample_file = "shakespeare_char_example.txt"
max_new_tokens = 2000

# Learning Model
batch_size = 64
block_size = 128
num_embeddings = 224
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning
learning_rate = 8e-4
max_iterations = 800
