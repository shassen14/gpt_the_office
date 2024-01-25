# TODO: organize and comment

# dataset
dataset_dir = 'data/shakespeare_char'
train_file = 'train.bin'
val_file = 'val.bin'
pkl_file = 'meta.pkl'

# params
param_dir = 'params'
pt_file = 'test_char_cfg.pt'

# generate sample
sample_dir = 'examples'
sample_file = 'shakespeare_char_example.txt'
max_new_tokens = 5000

# Learning Model
batch_size = 64
block_size = 256
num_embeddings = 448
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning
learning_rate = 3e-4
max_iterations = 2000
