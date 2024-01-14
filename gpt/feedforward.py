import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embeddings, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)