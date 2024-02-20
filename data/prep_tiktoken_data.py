"""
Prepare dataset for word/part-of-word/character-level language modeling. 
We map characters to ints. Will save train.bin, val.bin containing the ids,
and meta.pkl containing the encoder and decoder and some other related info.
"""

import data_config as dc
import utils

import os
import pickle
import requests
import tiktoken
import numpy as np

# TODO: vocab size is attached to tiktoken model. Make this configurable for
# different models
vocab_size = 50257
tiktoken_model = "gpt2"
is_tiktoken = True

# download the desired dataset
# obtain folder and file path
tiktoken_folder = dc.folder_name + "_" + tiktoken_model
input_folder_path = os.path.join(os.path.dirname(__file__), tiktoken_folder)
input_file_path = input_folder_path + "/script.txt"

# create folder directory if none exists
if not os.path.exists(input_folder_path):
    os.makedirs(input_folder_path)

# download file and write to txt if it doesn't exist
if not os.path.exists(input_file_path):
    with open(input_file_path, "w") as f:
        f.write(requests.get(dc.url).text)

with open(input_file_path, "r") as f:
    data = f.read()

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding(tiktoken_model)
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"vocab size: {vocab_size:,}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(input_folder_path + "/train.bin")
val_ids.tofile(input_folder_path + "/val.bin")

# save the meta information as well, to help us encode/decode later
meta = {
    "is_tiktoken": is_tiktoken,
    "tiktoken_model": tiktoken_model,
    "vocab_size": vocab_size,
}
with open(input_folder_path + "/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

utils.write_to_config_data(tiktoken_folder)
