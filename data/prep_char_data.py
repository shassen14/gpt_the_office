"""
Prepare dataset for character-level language modeling. We map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import data_config as dc

import os
import pickle
import requests
import numpy as np

# download the desired dataset
# obtain folder and file path
input_folder_path = os.path.join(os.path.dirname(__file__), dc.folder_name)
input_file_path = input_folder_path + '/' + dc.script_file_name

# create folder directory if none exists
if not os.path.exists(input_folder_path):
    os.makedirs(input_folder_path)

# download file and write to txt if it doesn't exist
if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        f.write(requests.get(dc.url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f'length of dataset in characters: {len(data):,}')

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print('all the unique characters:', ''.join(chars))
print(f'vocab size: {vocab_size:,}')

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f'train has {len(train_ids):,} tokens')
print(f'val has {len(val_ids):,} tokens')

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(input_folder_path + '/' + dc.train_file_name)
val_ids.tofile(input_folder_path + '/' + dc.val_file_name)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(input_folder_path + '/' + dc.pkl_file_name, 'wb') as f:
    pickle.dump(meta, f)

# For Shakespeare,
# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens