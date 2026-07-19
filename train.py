import pickle
import os
from model import Decoder, Config
import numpy as np
import torch

base_dir = os.path.dirname(__file__)

config : Config
config_path = os.path.join(base_dir, "config.pickle")
with open(config_path, 'rb') as f:
    config = pickle.load(f)
    print(config)
model = Decoder(config)
print(model)

# Apparently we save it as uint16 to save storage space locally, but then cast to int64.
# So the embedding() didn't work with even unit16, int16, but only with int64, and int32
# This is because the kernel is implemented for these standard types.
train_bin_path = os.path.join(base_dir, "train.bin")
with open(train_bin_path, 'r') as f:
    train_set = np.fromfile(f, dtype=np.uint16).astype(dtype=np.int64) # np.array(f.read())
    print(f"train set try1: {train_set[:10]}, size: {len(train_set)}")

train_set = np.fromfile(train_bin_path, dtype=np.uint16).astype(dtype=np.int64)
print(f"train set try2: {train_set[:10]}, size {len(train_set)}")

# WHY I'M KEEPING THIS!
# At this point, I ended up breaking the x into [num_iters, B, T], but we just need [B, T] and just stack it up for all the tensors.
# my initial idea was to limit the training runs with the num_iters, but although we would end up with a smaller train_set memory, which now thinking about it
# seems to be the right thing to do in prod, but we can always exit the loop, but wait. I think I'll keep this.
# Divide up the data into batches
num_batches = len(train_set) / config.batch_size
print(f"num_batches: {num_batches}")

# let's start with 100 batches

# for i in range(0, config.batch_size * 5, config.batch_size):
#     print(f"batch_no: {i//config.batch_size}, element: {i}")

#     # I'd like to feed (B, T), but currrently its just B. I need to divide it further
#     # We also need to randomize where we get it from

# divide up into chuncks of batch sizes and pick random instances
num_iters = 5
idx = torch.randint(0, len(train_set) - config.context_len, (num_iters, config.batch_size)) # num_iters, 

# splice [B, T+1] from the data such that T+1st element is the Y
# [num_iter, B, T] = X, [num_iter, B, 1] = Y
train_batch_list = [[train_set[idx[i,b]: idx[i,b] + config.context_len + 1] for b in range(config.batch_size)] for i in range(num_iters)]
train_batch_list = torch.tensor(np.array(train_batch_list)) # (num_iters, B, T)

print(f"Obtained the training batch: {train_batch_list.shape}")

model = Decoder(config)

# next part is to go through each iters, and pass just the 

for itr in range(num_iters):
    # pass it to the model
    x = train_batch_list[itr] # (B, T)
    print(f"input: {x.shape}, {x.dtype}")
    y_bar = model(x)

    print(f"received model output: {y_bar.shape}")

    pass
