import torch.nn as nn

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, num_embeddings, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)