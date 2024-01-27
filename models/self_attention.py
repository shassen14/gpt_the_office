import torch
import torch.nn as nn
from torch.nn import functional as F


class SingleHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.num_embeddings, cfg.head_size, bias=False)
        self.query = nn.Linear(cfg.num_embeddings, cfg.head_size, bias=False)
        self.value = nn.Linear(cfg.num_embeddings, cfg.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        weight = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)

        # perform the weightghted aggregation of the values
        out = weight @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiHead(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(cfg) for _ in range(cfg.num_heads)])
        self.proj = nn.Linear(cfg.head_size * cfg.num_heads, cfg.num_embeddings)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, cfg):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.num_embeddings, 4 * cfg.num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * cfg.num_embeddings, cfg.num_embeddings),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, cfg):
        # num_embeddings: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHead(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.num_embeddings)
        self.ln2 = nn.LayerNorm(cfg.num_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(nn.Module):
    """ " TODO: Description here"""

    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.block_size = cfg.block_size
        self.device_type = cfg.device_type
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.num_embeddings)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.num_embeddings)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.num_embeddings)  # final layer norm
        self.lm_head = nn.Linear(cfg.num_embeddings, vocab_size)

        # TODO: explain
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """TODO: what is this?"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device_type)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    @torch.no_grad()
    def generate2(self, idx):
        # idx is (B, T) array of indices in the current context

        # crop idx to the last block_size tokens
        idx_cond = idx[:, -self.block_size :]

        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # append sampled index to the running sequence
        idx = torch.cat((idx_cond, idx_next), dim=1)  # (B, T+1)

        return idx, idx_next
