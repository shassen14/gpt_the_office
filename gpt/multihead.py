import gpt.singlehead as sh

import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self,
                 block_size,
                 n_embeddings,
                 head_size,
                 num_heads,
                 dropout):
        super().__init__()
        self.heads = nn.ModuleList([sh.Attention(block_size, n_embeddings, head_size, dropout)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
