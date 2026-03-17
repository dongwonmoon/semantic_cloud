import torch
from torch import nn


class TinyTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        max_length: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.full((tokens.size(1), tokens.size(1)), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        encoded = self.decoder(hidden, mask=causal_mask)
        return self.output(encoded)
