"""
Hyperparameters for a "large" model
"""
# Learning Model
batch_size = 8
block_size = 128
num_embeddings = 64
num_heads = 6
num_layers = 8
dropout = 0.2

# Learning / Optimizer
max_iterations = 4000
eval_iterations = 200
warmup_iterations = int(float(max_iterations) * 0.1)  # warmup 10% of the time
decay_iterations = int(float(max_iterations) * 0.9)  # dacayed to min_lr after 90%
max_learning_rate = 3e-4
min_learning_rate = 1e-6
