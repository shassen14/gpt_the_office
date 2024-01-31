"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
from dataclasses import dataclass, field
from typing import List

import torch

# NOTE: Edit the following variables to choose initialization, data, model params
initialize = "start"
choose_data = "the_office_char"
choose_model = "medium"

if choose_data == "the_office_char":
    from .data import the_office_char as dt
elif choose_data == "the_office_gpt2":
    from .data import the_office_gpt2 as dt
else:
    print(f"data, {choose_data}, isn't valid. Please choose or add a valid one.")
    exit()

if choose_model == "small":
    from .model import small as mdl
elif choose_model == "medium":
    from .model import medium as mdl
elif choose_model == "large":
    from .model import large as mdl
else:
    print(f"model, {choose_model}, isn't valid. Please choose or add a valid one.")
    exit()

# Logging where we are initializing from, what data and model we are utilizing
print(f"Initializing: {initialize}")
print(f"Data: {choose_data}")
print(f"Model size: {choose_model}")


@dataclass
class Config:
    # fmt: off
    # Initialize
    initialize: str         = initialize

    # Dataset to utilize
    dataset_dir: str        = dt.dataset_dir
    train_file: str         = "train.bin"
    val_file: str           = "val.bin"
    pkl_file: str           = "meta.pkl"
    file_array: List[str]   = field(default_factory=lambda: [Config.train_file, Config.val_file])

    # Parameter Save/Load
    param_dir: str          = "params"
    pt_file: str            = dt.file_name + "_" + choose_model + ".pt"

    ## Generate sample texts
    sample_dir: str         = "examples"
    sample_file: str        = dt.file_name + "_" + choose_model + ".txt"
    max_new_tokens: int     = dt.max_new_tokens

    ############################## Paramaters #####################################
    # Model
    block_size: int         = mdl.block_size
    batch_size: int         = mdl.batch_size
    num_embeddings: int     = mdl.num_embeddings
    num_heads: int          = mdl.num_heads
    num_layers: int         = mdl.num_layers
    head_size: int          = num_embeddings // num_heads
    dropout: float          = mdl.dropout

    # Optimizer
    max_iterations: int     = dt.max_iterations
    eval_iterations: int    = dt.eval_iterations
    warmup_iterations: int  = dt.warmup_iterations
    decay_iterations: int   = dt.decay_iterations
    max_learning_rate: float= dt.max_learning_rate
    min_learning_rate: float= dt.min_learning_rate

    # get device type. get GPU or apple if possible
    device_type: str        = "cpu"
    # fmt: on
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"

    print("Device Type: " + device_type)
