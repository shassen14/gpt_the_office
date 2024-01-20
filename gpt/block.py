import gpt.multihead as mh
import gpt.feedforward as ff

import torch.nn as nn

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,
                 block_size,
                 num_embeddings,
                 head_size,
                 num_heads,
                 dropout):
        # num_embeddings: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = num_embeddings // num_heads
        self.sa = mh.Attention(block_size, num_embeddings, head_size, num_heads, dropout)
        self.ffwd = ff.FeedForward(num_embeddings, dropout)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
