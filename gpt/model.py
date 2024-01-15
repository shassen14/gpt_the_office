import gpt.block as b

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):
    """" TODO: Description here"""
    def __init__(self,
                 vocab_size,
                 num_layers,
                 block_size,
                 n_embeddings,
                 head_size,
                 num_heads,
                 dropout,
                 device_type: str):
        super().__init__()
        self.block_size = block_size
        self.device_type = device_type
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(*[b.Block(block_size, n_embeddings, head_size, num_heads, dropout) 
                                      for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embeddings) # final layer norm
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

        # TODO: explain
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ TODO: what is this?"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device_type)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss  

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx