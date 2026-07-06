import pickle
import os
from model import Decoder, Config
import numpy as np
import torch

base_dir = os.path.dirname(__file__) + "/"

config : Config
with open(base_dir + "config.pickle", 'rb') as f:
    config = pickle.load(f)
    print(config)
model = Decoder(config)
print(model)

with open(base_dir + "train.bin", 'r') as f:
    train_set = np.fromfile(f, dtype=np.uint16) # np.array(f.read())
    print(f"train set try1: {train_set[:10]}, size: {len(train_set)}")

train_set = np.fromfile(base_dir + "train.bin", dtype=np.uint16)
print(f"train set try2: {train_set[:10]}, size {len(train_set)}")

# Divide up the data into batches
num_batches = len(train_set) / config.batch_size
print(f"num_batches: {num_batches}")

# let's start with 100 batches

# for i in range(0, config.batch_size * 5, config.batch_size):
#     print(f"batch_no: {i//config.batch_size}, element: {i}")

#     # I'd like to feed (B, T), but currrently its just B. I need to divide it further
#     # We also need to randomize where we get it from

# divide up into chuncks of batch sizes and pick random instances
num_iters = 100
idx = torch.randint(0, len(train_set), (num_iters, config.batch_size)) # num_iters, 

# splice [B, T+1] from the data such that T+1st element is the Y
# [num_iter, B, T] = X, [num_iter, B, 1] = Y
x = torch.stack((train_set[i] for i in idx))