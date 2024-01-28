# TODO: organize and comment
import os

# Resume or Start
initialize = "resume"

# dataset
dataset = "the_office"
dataset_type = "char"
cfg_file_name = os.path.basename(__file__)[:-3]
dataset_dir = "data/" + dataset + "_" + dataset_type

# params
pt_file = dataset + "_" + cfg_file_name + ".pt"

# generate sample
sample_file = dataset + "_" + dataset_type + "" + ".txt"
max_new_tokens = 2000

# Learning Model
batch_size = 16
block_size = 256
num_embeddings = 384
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning / Optimizer
max_iterations = 8000
eval_iterations = 200
max_learning_rate = 3e-4
min_learning_rate = 1e-5
