import torch
from torch import nn


class GRUBaselineClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(tokens)
        encoded, _ = self.encoder(embeddings)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        features = torch.cat([mean_pool, max_pool], dim=-1)
        return self.classifier(features)
