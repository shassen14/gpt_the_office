# TODO: organize and comment
import os

# Resume or Start
initialize = "start"

# dataset
cfg_file_name = os.path.basename(__file__)[:-3]
dataset_dir = "data/" + cfg_file_name

# params
pt_file = cfg_file_name + ".pt"

# generate sample
sample_file = cfg_file_name + "" + ".txt"
max_new_tokens = 2000

# Learning Model
batch_size = 16
block_size = 128
num_embeddings = 128
num_heads = 6
num_layers = 6
dropout = 0.2

# Learning / Optimizer
max_iterations = 8000
eval_iterations = 200
warmup_iterations = int(float(max_iterations) * 0.1)  # warmup 10% of the time
decay_iterations = int(float(max_iterations) * 0.9)  # dacayed to min_lr after 90%
max_learning_rate = 3e-4
min_learning_rate = 1e-5
