# TODO: organize and comment
import os

# dataset folder name and this file should be the same
file_name = os.path.basename(__file__)[:-3]
dataset_dir = "data/" + file_name

# generate sample
max_new_tokens = 2000

# Learning / Optimizer
max_iterations = 10000
eval_iterations = 200
warmup_iterations = int(float(max_iterations) * 0.1)  # warmup 10% of the time
decay_iterations = int(float(max_iterations) * 0.9)  # dacayed to min_lr after 90%
max_learning_rate = 3e-4
min_learning_rate = 1e-5
