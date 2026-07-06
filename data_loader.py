# Feed into the dataloader for traiing/validation
# Get 90% for traiing and the rest for validation.

import requests
import os
import numpy as np
import pickle
from model import Config
import json

class FileManager:
    def __init__(self, file_name="input.txt"):
        self.file_path = os.path.dirname(__file__) + "/" + file_name
    
    def save_to(self, file_path):
        with open(file_path, 'w') as f:
            print(self.r.text, file=f)

    def download_input_file(self):
        # apparently __file__ is available as the path to this current file!
        if os.path.exists(self.file_path):
            print(f"The file path is {self.file_path} and it exists")
            return
    
        print(f"The file path is {self.file_path} and it doesn't exist")

        data_url = "https://sherlock-holm.es/stories/plain-text/cano.txt"
        self.r = requests.get(data_url)
        self.r.raise_for_status()

        print(f"Downloaded with size {len(self.r.text)}")
    
    def read_file(self):
        with open(self.file_path, "r") as r:
            return ''.join(r.readlines())

file_manager = FileManager()

# Extract 
# vocab = 'abcdefghijklmnopqrstuvwxyz ' # this was a great starting point! But we could just get the entire text instead!
vocab = list(set(file_manager.read_file()))
print(f"vocab size: {len(vocab)}, vocab chars: {vocab[:10]}")
stoi = lambda x : vocab.index(x)
itos = lambda x : vocab[x]
print(f"a->{stoi('a')}, b->{stoi('b')}")
print(f"25->{itos(25)}, 3->{itos(3)}")
encode = lambda x : [stoi(j) for j in x]# [1,2,4,3]
decode = lambda x : ''.join([itos(k) for k in x])
print(f"Encoding a {encode('an apple')}")
print(f"Decoding \"An apple on the tree\", {encode("An apple on the tree")} and {decode(encode("An apple on the tree "))}")

# download_input_file()
data = file_manager.read_file()
train_set = encode(data[:int(0.9*len(data))])
val_set = encode(data[int(0.9*len(data)):])

print(f"Train set has {len(train_set)} tokens, val set has {len(val_set)} tokens")

train_set_bin = np.array(train_set, dtype=np.uint16)
val_set_bin = np.array(val_set, dtype=np.uint16)

print(f"saving trainset {train_set_bin[:10]} \n and val set {val_set_bin[:10]}")
train_set_bin.tofile(os.path.dirname(__file__) + "/train.bin")
val_set_bin.tofile(os.path.dirname(__file__) + "/val.bin")

config = Config(vocab_size=len(vocab))

with open(os.path.dirname(__file__) + "/config.pickle", mode="wb") as f:
    pickle.dump(config, f, pickle.HIGHEST_PROTOCOL) 

# with open(os.path.dirname(__file__) + "/config.json", mode='w') as f:
#     json.dump(asdict(config), f)

# with open("Just_a_new_file.txt", 'w') as f: # without the __file__, it just writes wherever you call this from.
#     print('Where is this written', file=f)