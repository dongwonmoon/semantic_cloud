import torch
import torch.nn.functional as F
from torch import nn


class CFRMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_clouds: int,
        hidden_dim: int = 128,
        embedding_dim: int | None = None,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_clouds = num_clouds
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        embedding_dim = embedding_dim or hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.local_decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        controller_in_dim = 2 * hidden_dim + 4
        self.controller = nn.Sequential(
            nn.Linear(controller_in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.update_gate_proj = nn.Linear(hidden_dim, num_clouds)
        self.assign_proj = nn.Linear(hidden_dim, num_clouds)
        self.novelty_proj = nn.Linear(hidden_dim, 1)
        self.relax_proj = nn.Linear(hidden_dim, 1)
        self.center_candidate_proj = nn.Linear(hidden_dim, num_clouds * hidden_dim)
        self.spread_candidate_proj = nn.Linear(hidden_dim, num_clouds)
        self.mass_delta_proj = nn.Linear(hidden_dim, num_clouds)
        self.attractor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def summarize_state(self, centers: torch.Tensor, spreads: torch.Tensor, masses: torch.Tensor):
        precision = 1.0 / (spreads + 1e-4)
        scores = masses + torch.log(precision + 1e-4)
        alpha = torch.softmax(scores, dim=-1)
        core = torch.sum(alpha.unsqueeze(-1) * centers, dim=1)
        uncertainty = torch.sum(alpha * spreads, dim=-1, keepdim=True)
        sq_dist = (centers - core.unsqueeze(1)).pow(2).mean(dim=-1)
        diversity = torch.sum(alpha * sq_dist, dim=-1, keepdim=True)
        energy = torch.logsumexp(masses, dim=-1, keepdim=True)
        entropy = -torch.sum(alpha * torch.log(alpha.clamp_min(1e-8)), dim=-1, keepdim=True)
        return core, uncertainty, diversity, energy, entropy, alpha

    def cloud_interaction(self, centers: torch.Tensor, spreads: torch.Tensor, masses: torch.Tensor):
        d2 = torch.cdist(centers, centers, p=2).pow(2)
        scale = spreads.unsqueeze(1) + spreads.unsqueeze(2) + 1e-4
        compat = -d2 / scale + masses.unsqueeze(1)
        mixing = torch.softmax(compat, dim=-1)
        mixed_centers = torch.matmul(mixing, centers)
        mixed_spreads = torch.sum(mixing * spreads.unsqueeze(1), dim=-1)
        mixed_masses = torch.sum(mixing * masses.unsqueeze(1), dim=-1)
        return mixed_centers, mixed_spreads, mixed_masses

    def forward(self, tokens: torch.Tensor, return_state: bool = False):
        batch_size, seq_len = tokens.shape
        device = tokens.device
        mask = (tokens != self.pad_idx).float()
        local_hidden, _ = self.local_decoder(self.embedding(tokens))

        centers = torch.zeros(batch_size, self.num_clouds, self.hidden_dim, device=device)
        spreads = torch.ones(batch_size, self.num_clouds, device=device)
        masses = torch.zeros(batch_size, self.num_clouds, device=device)
        logits_history = []
        novelty_history = []
        core_history = []

        for t in range(seq_len):
            valid = mask[:, t : t + 1]
            local_t = local_hidden[:, t, :]
            core, uncertainty, diversity, energy, entropy, alpha = self.summarize_state(
                centers, spreads, masses
            )
            ctrl = self.controller(
                torch.cat([local_t, core, uncertainty, diversity, energy, entropy], dim=-1)
            )
            gate = torch.sigmoid(self.update_gate_proj(ctrl)) * valid
            assign = torch.softmax(self.assign_proj(ctrl), dim=-1)
            novelty = torch.sigmoid(self.novelty_proj(ctrl)) * valid
            relax = torch.sigmoid(self.relax_proj(ctrl)) * valid
            novelty_history.append(novelty)

            candidate_centers = self.center_candidate_proj(ctrl).view(
                batch_size, self.num_clouds, self.hidden_dim
            )
            candidate_spreads = F.softplus(self.spread_candidate_proj(ctrl)) + 1e-4
            mass_delta = torch.tanh(self.mass_delta_proj(ctrl))

            step_strength = gate * assign
            centers = centers + step_strength.unsqueeze(-1) * (candidate_centers - centers)
            spreads = spreads + step_strength * (candidate_spreads - spreads)
            masses = masses + step_strength * mass_delta

            attractor = self.attractor_proj(ctrl).unsqueeze(1)
            centers = centers + 0.10 * novelty.unsqueeze(-1) * (attractor - centers)

            mixed_centers, mixed_spreads, mixed_masses = self.cloud_interaction(
                centers, spreads, masses
            )
            centers = (1.0 - relax.unsqueeze(-1)) * centers + relax.unsqueeze(-1) * mixed_centers
            spreads = (1.0 - relax) * spreads + relax * mixed_spreads
            masses = (1.0 - relax) * masses + relax * mixed_masses

            core, uncertainty, diversity, energy, entropy, alpha = self.summarize_state(
                centers, spreads, masses
            )
            strongest_idx = torch.argmax(alpha, dim=-1)
            strongest_center = centers[torch.arange(batch_size, device=device), strongest_idx]
            core_history.append(core)
            step_features = torch.cat(
                [local_t, core, strongest_center, uncertainty, diversity, energy, entropy],
                dim=-1,
            )
            logits_history.append(self.output(step_features))

        logits = torch.stack(logits_history, dim=1)
        if not return_state:
            return logits

        return {
            "logits": logits,
            "core": torch.stack(core_history, dim=1),
            "centers": centers,
            "spreads": spreads,
            "masses": masses,
            "novelty": torch.stack(novelty_history, dim=1),
        }
