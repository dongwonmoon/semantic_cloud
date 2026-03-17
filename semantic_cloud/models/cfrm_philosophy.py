import torch
import torch.nn.functional as F
from torch import nn


class CFRMPhilosophyClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_clouds: int,
        hidden_dim: int = 128,
        pad_idx: int = 0,
        sparse_reconfiguration: bool = False,
        reconfiguration_interval: int = 1,
        novelty_threshold: float = 0.0,
        always_apply_attractor: bool = False,
        interaction_topk: int | None = None,
    ) -> None:
        super().__init__()
        self.num_clouds = num_clouds
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.sparse_reconfiguration = sparse_reconfiguration
        self.reconfiguration_interval = max(1, reconfiguration_interval)
        self.novelty_threshold = novelty_threshold
        self.always_apply_attractor = always_apply_attractor
        self.interaction_topk = interaction_topk

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.token_norm = nn.LayerNorm(hidden_dim)

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

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
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
        if self.interaction_topk is not None and self.interaction_topk < compat.size(-1):
            topk_values, topk_indices = torch.topk(compat, k=self.interaction_topk, dim=-1)
            masked = torch.full_like(compat, float("-inf"))
            compat = masked.scatter(-1, topk_indices, topk_values)
        mixing = torch.softmax(compat, dim=-1)

        mixed_centers = torch.matmul(mixing, centers)
        mixed_spreads = torch.sum(mixing * spreads.unsqueeze(1), dim=-1)
        mixed_masses = torch.sum(mixing * masses.unsqueeze(1), dim=-1)
        return mixed_centers, mixed_spreads, mixed_masses

    def forward(self, tokens: torch.Tensor, return_state: bool = False):
        batch_size, seq_len = tokens.shape
        device = tokens.device
        mask = (tokens != self.pad_idx).float()
        token_emb = self.token_norm(self.embedding(tokens))

        centers = torch.zeros(batch_size, self.num_clouds, self.hidden_dim, device=device)
        spreads = torch.ones(batch_size, self.num_clouds, device=device)
        masses = torch.zeros(batch_size, self.num_clouds, device=device)
        novelty_history = []
        reconfiguration_mask = []
        attractor_mask = []

        for t in range(seq_len):
            valid = mask[:, t : t + 1]
            token_t = token_emb[:, t, :]
            core, uncertainty, diversity, energy, entropy, alpha = self.summarize_state(
                centers, spreads, masses
            )
            control_input = torch.cat(
                [token_t, core, uncertainty, diversity, energy, entropy],
                dim=-1,
            )
            ctrl = self.controller(control_input)

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

            should_reconfigure = True
            if self.sparse_reconfiguration:
                interval_hit = (t % self.reconfiguration_interval) == 0
                novelty_hit = bool((novelty.max().item()) >= self.novelty_threshold)
                should_reconfigure = interval_hit and novelty_hit
            reconfiguration_mask.append(float(should_reconfigure))

            should_apply_attractor = should_reconfigure or self.always_apply_attractor
            attractor_mask.append(float(should_apply_attractor))

            if should_apply_attractor:
                attractor = self.attractor_proj(ctrl).unsqueeze(1)
                centers = centers + 0.10 * novelty.unsqueeze(-1) * (attractor - centers)

            if should_reconfigure:
                mixed_centers, mixed_spreads, mixed_masses = self.cloud_interaction(
                    centers, spreads, masses
                )
                centers = (1.0 - relax.unsqueeze(-1)) * centers + relax.unsqueeze(-1) * mixed_centers
                spreads = (1.0 - relax) * spreads + relax * mixed_spreads
                masses = (1.0 - relax) * masses + relax * mixed_masses

            precision = 1.0 / (spreads + 1e-4)
            condense_alpha = torch.softmax(masses + torch.log(precision + 1e-4), dim=-1)
            spreads = spreads * (1.0 - 0.05 * condense_alpha * valid) + 1e-4

        core, uncertainty, diversity, energy, entropy, alpha = self.summarize_state(
            centers, spreads, masses
        )
        strongest_idx = torch.argmax(alpha, dim=-1)
        strongest_center = centers[torch.arange(batch_size, device=device), strongest_idx]
        features = torch.cat(
            [core, strongest_center, uncertainty, diversity, energy, entropy],
            dim=-1,
        )
        logits = self.classifier(features)

        if not return_state:
            return logits

        return {
            "logits": logits,
            "centers": centers,
            "spreads": spreads,
            "masses": masses,
            "alpha": alpha,
            "core": core,
            "strongest_center": strongest_center,
            "uncertainty": uncertainty,
            "diversity": diversity,
            "energy": energy,
            "entropy": entropy,
            "novelty": torch.stack(novelty_history, dim=1),
            "reconfiguration_count": int(sum(reconfiguration_mask)),
            "reconfiguration_mask": reconfiguration_mask,
            "attractor_count": int(sum(attractor_mask)),
            "attractor_mask": attractor_mask,
        }
