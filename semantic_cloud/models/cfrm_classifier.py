import torch
from torch import nn


class CFRMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_clouds: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_clouds = num_clouds
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.center_proj = nn.Linear(hidden_dim, num_clouds * hidden_dim)
        self.spread_proj = nn.Linear(hidden_dim, num_clouds)
        self.weight_proj = nn.Linear(hidden_dim, num_clouds)
        self.classifier = nn.Linear(num_clouds * (hidden_dim + 2), num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(tokens)
        encoded, _ = self.encoder(embeddings)

        batch_size = encoded.size(0)
        centers = torch.zeros(
            batch_size,
            self.num_clouds,
            self.hidden_dim,
            device=tokens.device,
        )
        spreads = torch.ones(batch_size, self.num_clouds, device=tokens.device)
        weights = torch.zeros(batch_size, self.num_clouds, device=tokens.device)

        for step_hidden in encoded.unbind(dim=1):
            center_delta = self.center_proj(step_hidden).view(
                batch_size, self.num_clouds, self.hidden_dim
            )
            spread_delta = torch.sigmoid(self.spread_proj(step_hidden))
            weight_delta = self.weight_proj(step_hidden)

            centers = 0.85 * centers + 0.15 * center_delta
            spreads = 0.85 * spreads + 0.15 * spread_delta
            weights = weights + weight_delta

        normalized_weights = torch.softmax(weights, dim=-1)
        summary = torch.cat(
            [
                centers,
                spreads.unsqueeze(-1),
                normalized_weights.unsqueeze(-1),
            ],
            dim=-1,
        )
        flat_summary = summary.reshape(batch_size, -1)
        return self.classifier(flat_summary)
