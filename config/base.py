"""
Choose configuration to use
 
block_size: maximum context length for the model to utilize
batch_size:
"""
from dataclasses import dataclass, field
from typing import List

import torch

choose_config = "test_char_cfg"

if choose_config == "test_char_cfg":
    import config.test_char_cfg as cfg
elif choose_config == "test_tiktoken_cfg":
    import config.test_tiktoken_cfg as cfg
elif choose_config == "small_cfg":
    import config.small_cfg as cfg
else:
    print(f"config, {choose_config}, isn't valid. Please choose or add a valid one.")
    exit()


@dataclass
class Config:
    # fmt: off
    # Dataset to utilize
    dataset_dir: str        = cfg.dataset_dir
    train_file: str         = "train.bin"
    val_file: str           = "val.bin"
    pkl_file: str           = "meta.pkl"
    file_array: List[str]   = field(default_factory=lambda: [Config.train_file, Config.val_file])

    # Parameter Save/Load
    param_dir: str          = cfg.param_dir
    pt_file: str            = cfg.pt_file

    ## Generate sample texts
    sample_dir: str         = cfg.sample_dir
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

    learning_rate: float    = cfg.learning_rate
    max_iterations: int     = cfg.max_iterations
    eval_iterations: int    = int(max_iterations / 10)

    # get device type. get GPU or apple if possible
    device_type: str        = "cpu"
    # fmt: on

    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"

    print("Device Type: " + device_type)
