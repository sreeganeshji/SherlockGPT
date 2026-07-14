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
        # self.lin_to_head = nn.Linear(config.embd_dim, config.embd_dim * config.num_heads, config.bias)

        # attention
        self.num_heads = config.num_heads
        # self.lin_kqv = nn.Linear(config.embd_dim * config.num_heads, config.embd_dim * config.num_heads * 3, config.bias)
        # NOTE: In andrej's implementation, they weren't scaling out by num_heads. They ensured that the embd dim ws divisible by num_heads and just capped at that.
        # Consulting

        # I'll be dividing up the heads from the num_embd
        assert self.embd_dim % config.num_heads == 0
        # (B, T, C/num_heads) but do I need num_heads separate linear elements? Do I need to initialize a different nn.Linear() for each of them? The reason I ask is that
        # these would be storing and learning the weights, and if I share tese elements, would that mean that they wouldn't learn differnt weights for say different heads?,
        # Or have different weights fro k, q, v? etc.
        '''
        nanoGPT does this slightly differently. Yes, it is true that when you initialize a linear element, that would map to a specivc set of 
        weights which need to be trained for one flow. Like you can't have lin(), and separately pass k, v, q = lin(x), lin(x), lin(x). That's an extreme.
        But how nanoGPT handles it is it has one lin() which maps from (C -> 3*C) and it ensures that C % num_heads == 0. So we're essentially just 
        viewing the new output as a different sized embd per head with the tensor.view().
        '''

        self.lin_kqv = nn.Linear(config.embd_dim, config.embd_dim * 3, bias=config.bias)


    def forward(self, x: torch.Tensor):
        # #embedding
        # x = self.embd(x)
        # x = self.lin_to_head(x) # (B, T, embd_dim * num_heads)
        # print(f"shape {x.shape}")
        # # We need to split it into three vectors. I'm guessing there's the view() function?
        # # to be able to split this into three, we need to have embeddings that'll expand this into three.
        # k, q, v = x.split(self.num_heads, 2) # (B, T, embd_dim). This should happen within each head.
        # print(f"split x{x.shape} in to k {k.shape}, q{q.shape} and v{v.shape}")

        x = self.embd(x) # (B, T) -> (B, T, C)
        kqv_combined = self.lin_kqv(x) # (B, T, 3*C)
        # k = (B, T, C/num_heads)
        k, q, v = kqv_combined.split(3, dim=2) # (B, T, C)
        # extract out num_heads into another dim

        pass

