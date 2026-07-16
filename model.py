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
        self.embd_dim = config.embd_dim
        # self.lin_kqv = nn.Linear(config.embd_dim * config.num_heads, config.embd_dim * config.num_heads * 3, config.bias)
        # NOTE: In andrej's implementation, they weren't scaling out by num_heads. They ensured that the embd dim ws divisible by num_heads and just capped at that.
        # Consulting

        # I'll be dividing up the heads from the num_embd
        assert config.embd_dim % config.num_heads == 0
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
        (k, q, v) = kqv_combined.split(self.embd_dim, dim=2) # (B, T, C)
        # extract out num_heads into another dim
        print(f"K dims {k.shape}")
        '''
        # Next step is to reshape into number of heads (B, T, C/head, head)
        # But we'd like it to use broadcasting
        So it might work better if we share the common factors at the beginning.
        (B, head, T, E) * (B, head, E, T) then we'll get TxT

        7/15
        Let's divide up the large array into three heads. We need too view() into (B, T, nh, hs), then
        transpose it to (B, nh, T, hs) this is because we want to retain the temporal data contiguousness.
        performing view() directly into (B, nh, T, hs) would corrupt this and would send out random temporal
        bits to random places.
        '''
        (B, T, C) = k.shape
        kqv_combined : torch.Tensor
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, T, nh, ns) -> (B, nh, T, ns)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, T, nh, ns) -> (B, nh, T, ns)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, T, nh, ns) -> (B, nh, T, ns)

        temp = (q @ k.transpose(3, 2))

        print(f"shape of (q@k.t): {temp.shape}")
        print(f"values at [0,0,:3, :3] {temp[0, 0, :3, :3]}")

        #normalizing
        n_dim = C // self.num_heads
        norm_factor = torch.sqrt(torch.tensor(n_dim))
        temp_n = temp / norm_factor # if we don't do this, the softmax would be spiky
        print(f"after normalizing with {norm_factor}: {temp_n[0, 0, :3, :3]} \n a factor of {(temp/temp_n)[0, 0, :3, :3]}")
        
        '''
        We need to get the probabilites for each query along the query dims. But we 
        apply the decoder triangular matrix. We have TxT matrix. Now multiply upper triangular matrix
        so that it can only see itself and the ones to the right
        1 1 1
        0 1 1
        0 0 1

        Then we can also get the softmax along each row by applying a normal dist that wouldn't sum to 1, but then 
        we apply the softmax. Because (q@k.t)/sqrt(ndim) would be a T * T matrix that doesn't necessarily sum to anything
        Even if we create a matrix that if summed along each row would lead to 1, that doesn't guarantee that you could 
        sum to 1.
        '''

        pass

