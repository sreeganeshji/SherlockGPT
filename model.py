import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size : int
    batch_size: int = 32
    context_len: int = 8
    embd_dim: int = 8
    num_heads: int = 4
    bias : bool = False

class Decoder(nn.Module):
    def __init__(self, config:Config):
        super().__init__()

        #embedding
        self.embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.lin_to_head = nn.Linear(config.embd_dim, config.embd_dim * config.num_heads, config.bias)

    def forward(self, x):
        #embedding
        x = self.embd(x)
        x = self.lin_to_head(x) # (B, T, embd_dim * num_heads)
        print(f"shape {x.shape}")

        pass

