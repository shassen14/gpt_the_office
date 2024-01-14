import gpt.multihead as mh
import gpt.feedforward as ff

import torch
import torch.nn as nn

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,
                 block_size,
                 n_embeddings,
                 head_size,
                 num_heads,
                 dropout):
        # n_embeddings: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embeddings // num_heads
        self.sa = mh(block_size, n_embeddings, head_size, num_heads, dropout)
        self.ffwd = ff(n_embeddings, dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
