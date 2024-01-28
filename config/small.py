# TODO: organize and comment
import os

# Resume or Start
initialize = "start"

# dataset
dataset = "the_office"
dataset_type = "char"
cfg_file_name = os.path.basename(__file__)[:-3]
dataset_dir = "data/" + dataset + "_" + dataset_type

# params
pt_file = dataset + "_" + cfg_file_name + ".pt"

# generate sample
sample_file = dataset + "_" + dataset_type + "" + ".txt"
max_new_tokens = 1000

# Model
batch_size = 4
block_size = 8
num_embeddings = 32
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning
max_iterations = 8000
eval_iterations = 200
max_learning_rate = 3e-3
min_learning_rate = 1e-5
