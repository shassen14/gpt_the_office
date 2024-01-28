"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
from dataclasses import dataclass, field
from typing import List

import torch

choose_config = "test_tiktoken"

if choose_config == "test_char":
    import config.test_char as cfg
elif choose_config == "test_tiktoken":
    import config.test_tiktoken as cfg
elif choose_config == "small":
    import config.small as cfg
else:
    print(f"config, {choose_config}, isn't valid. Please choose or add a valid one.")
    exit()


@dataclass
class Config:
    # fmt: off
    # Initialize
    initialize: str         = cfg.initialize

    # Dataset to utilize
    dataset_dir: str        = cfg.dataset_dir
    train_file: str         = "train.bin"
    val_file: str           = "val.bin"
    pkl_file: str           = "meta.pkl"
    file_array: List[str]   = field(default_factory=lambda: [Config.train_file, Config.val_file])

    # Parameter Save/Load
    param_dir: str          = "params"
    pt_file: str            = cfg.pt_file

    ## Generate sample texts
    sample_dir: str         = "examples"
    sample_file: str        = cfg.sample_file
    max_new_tokens: int     = cfg.max_new_tokens

    ############################## Paramaters #####################################
    block_size: int         = cfg.block_size
    batch_size: int         = cfg.batch_size
    num_embeddings: int     = cfg.num_embeddings
    num_heads: int          = cfg.num_heads
    num_layers: int         = cfg.num_layers
    head_size: int          = num_embeddings // num_heads
    dropout: float          = cfg.dropout

    max_iterations: int     = cfg.max_iterations
    eval_iterations: int    = cfg.eval_iterations
    max_learning_rate: float= cfg.max_learning_rate


    # get device type. get GPU or apple if possible
    device_type: str        = "cpu"
    # fmt: on
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"

    print("Device Type: " + device_type)
