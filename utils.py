import os
import pickle

import numpy as np
import torch

def get_batch(dataset_folder: str,
              dataset_type: str,
              block_size: int,
              batch_size: int,
              device_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Obtain file path
    # TODO: change if reorganizing files/directories
    file_path = get_file_path(dataset_folder, dataset_type)
    
    # save data
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    # random block chunk generator TODO: figure out what is happening here
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device_type, non_blocking=True), y.pin_memory().to(device_type, non_blocking=True)
    else:
        x, y = x.to(device_type), y.to(device_type)

    return x, y

###############################################################################

def get_folder_path(folder: str) -> str:
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

###############################################################################

def get_file_path(folder: str,
                  file: str) -> str:
    file_path = get_folder_path(folder) + '/' + file
    return file_path

