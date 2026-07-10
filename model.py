import torch.nn as nn
import torch
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

        # attention
        self.num_heads = config.num_heads
        self.lin_kqv = nn.Linear(config.embd_dim * config.num_heads, config.embd_dim * config.num_heads * 3, config.bias)
        # NOTE: In andrej's implementation, they weren't scaling out by num_heads. They ensured that the embd dim ws divisible by num_heads and just capped at that.
        # Consulting 

    def forward(self, x: torch.Tensor):
        #embedding
        x = self.embd(x)
        x = self.lin_to_head(x) # (B, T, embd_dim * num_heads)
        print(f"shape {x.shape}")
        # We need to split it into three vectors. I'm guessing there's the view() function?
        # to be able to split this into three, we need to have embeddings that'll expand this into three.
        k, q, v = x.split(self.num_heads, 2) # (B, T, embd_dim). This should happen within each head.
        print(f"split x{x.shape} in to k {k.shape}, q{q.shape} and v{v.shape}")

        pass

