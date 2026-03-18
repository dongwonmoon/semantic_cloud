import math

import torch
from torch import nn
import torch.nn.functional as F


def topk_shrink(x: torch.Tensor, lam: float, k: int) -> torch.Tensor:
    shrunk = torch.sign(x) * F.relu(torch.abs(x) - lam)
    if k <= 0 or k >= shrunk.size(-1):
        return shrunk
    _, indices = torch.topk(torch.abs(shrunk), k=k, dim=-1)
    mask = torch.zeros_like(shrunk, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    return torch.where(mask, shrunk, torch.zeros_like(shrunk))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(x.dtype)
    num = (x * mask.unsqueeze(-1)).sum(dim=dim)
    den = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return num / den


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    masked_x = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
    out = masked_x.max(dim=dim).values
    fully_masked = ~mask.any(dim=dim)
    if fully_masked.any():
        out = out.masked_fill(fully_masked.unsqueeze(-1), 0.0)
    return out


def chunk_mean_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, steps, hidden_dim = x.shape
    chunks = math.ceil(steps / chunk_size)
    pad_len = chunks * chunk_size - steps
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        mask = F.pad(mask, (0, pad_len), value=False)
    x = x.view(batch_size, chunks, chunk_size, hidden_dim)
    mask = mask.view(batch_size, chunks, chunk_size)
    mask_f = mask.to(x.dtype)
    denom = mask_f.sum(dim=2, keepdim=True).clamp_min(1.0)
    chunk_x = (x * mask_f.unsqueeze(-1)).sum(dim=2) / denom
    chunk_mask = mask.any(dim=2)
    return chunk_x, chunk_mask


class LocalPhraseEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.in_norm = nn.LayerNorm(model_dim)
        self.conv1 = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=5, padding=2)
        self.out_proj = nn.Linear(2 * model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.in_norm(x)
        y = x.transpose(1, 2)
        y1 = F.glu(self.conv1(y), dim=1)
        y2 = F.glu(self.conv2(y), dim=1)
        y = torch.cat([y1, y2], dim=1).transpose(1, 2)
        y = self.out_proj(y)
        y = self.dropout(y)
        return self.out_norm(x + y)


class SparseSemanticFieldCore(nn.Module):
    def __init__(
        self,
        model_dim: int,
        code_dim: int,
        topk: int = 8,
        shrink_lambda: float = 0.05,
        condense_rho: float = 0.90,
        dict_lr: float = 0.03,
        use_reconfiguration: bool = True,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.code_dim = code_dim
        self.topk = topk
        self.shrink_lambda = shrink_lambda
        self.condense_rho = condense_rho
        self.dict_lr = dict_lr
        self.use_reconfiguration = use_reconfiguration

        self.base_dictionary = nn.Parameter(torch.randn(model_dim, code_dim) * 0.02)
        self.code_from_u = nn.Linear(model_dim, code_dim, bias=False)
        self.code_from_c = nn.Linear(code_dim, code_dim, bias=False)
        self.candidate_from_u = nn.Linear(model_dim, model_dim * code_dim)
        self.candidate_from_c = nn.Linear(code_dim, model_dim * code_dim)
        self.reconfig_gate = nn.Linear(model_dim + code_dim + 1, 1)
        self.u_norm = nn.LayerNorm(model_dim)
        self.c_norm = nn.LayerNorm(code_dim)

    def init_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        dictionary = self.base_dictionary.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        dictionary = F.normalize(dictionary, dim=1)
        codes = torch.zeros(batch_size, self.code_dim, device=device)
        return {"D": dictionary, "c": codes}

    def infer_code(self, u_t: torch.Tensor, c_prev: torch.Tensor) -> torch.Tensor:
        logits = self.code_from_u(self.u_norm(u_t)) + self.code_from_c(self.c_norm(c_prev))
        return topk_shrink(logits, lam=self.shrink_lambda, k=self.topk)

    def reconstruct(self, dictionary: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return torch.bmm(dictionary, code.unsqueeze(-1)).squeeze(-1)

    def candidate_dictionary(self, u_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        batch_size = u_t.size(0)
        candidate = self.candidate_from_u(u_t) + self.candidate_from_c(c_t)
        candidate = candidate.view(batch_size, self.model_dim, self.code_dim)
        return F.normalize(candidate, dim=1)

    def step(
        self,
        u_t: torch.Tensor,
        state: dict[str, torch.Tensor],
        valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        dictionary = state["D"]
        c_prev = state["c"]
        a_t = self.infer_code(u_t, c_prev)
        u_hat = self.reconstruct(dictionary, a_t)
        residual = u_t - u_hat
        c_new = self.condense_rho * c_prev + (1.0 - self.condense_rho) * a_t
        c_t = valid * c_new + (1.0 - valid) * c_prev

        delta_D = torch.bmm(residual.unsqueeze(-1), a_t.unsqueeze(1))
        d_local = F.normalize(dictionary + self.dict_lr * delta_D, dim=1)

        if self.use_reconfiguration:
            err = residual.norm(dim=-1, keepdim=True)
            gate_in = torch.cat([u_t, c_t, err], dim=-1)
            gate = torch.sigmoid(self.reconfig_gate(gate_in))
            d_cand = self.candidate_dictionary(u_t, c_t)
            d_next = (1.0 - gate).unsqueeze(-1) * d_local + gate.unsqueeze(-1) * d_cand
            d_next = F.normalize(d_next, dim=1)
        else:
            gate = torch.zeros(u_t.size(0), 1, device=u_t.device, dtype=u_t.dtype)
            d_next = d_local

        d_next = valid.unsqueeze(-1) * d_next + (1.0 - valid).unsqueeze(-1) * dictionary
        z_t = torch.bmm(d_next, c_t.unsqueeze(-1)).squeeze(-1)
        return {"D": d_next, "c": c_t}, {
            "a_t": a_t,
            "residual": residual,
            "error": residual.norm(dim=-1, keepdim=True),
            "z_t": z_t,
            "gate": gate,
        }


class HierarchicalSparseFieldClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        model_dim: int = 128,
        code_dim: int = 32,
        topk: int = 8,
        chunk_size: int = 4,
        pad_idx: int = 0,
        dropout: float = 0.1,
        use_reconfiguration: bool = True,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.chunk_size = chunk_size
        self.local_encoder = LocalPhraseEncoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            pad_idx=pad_idx,
            dropout=dropout,
        )
        self.field = SparseSemanticFieldCore(
            model_dim=model_dim,
            code_dim=code_dim,
            topk=topk,
            use_reconfiguration=use_reconfiguration,
        )
        feat_dim = model_dim + code_dim + model_dim + model_dim + 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 2 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor, return_state: bool = False):
        token_mask = tokens != self.pad_idx
        local_h = self.local_encoder(tokens)
        surface_mean = masked_mean(local_h, token_mask, dim=1)
        surface_max = masked_max(local_h, token_mask, dim=1)
        chunk_h, chunk_mask = chunk_mean_pool(local_h, token_mask, self.chunk_size)
        batch_size, chunks, _ = chunk_h.shape
        state = self.field.init_state(batch_size, tokens.device)
        codes = []
        gates = []
        last_z = None
        last_c = None
        last_error = None
        last_gate = None

        for chunk_index in range(chunks):
            u_j = chunk_h[:, chunk_index, :]
            valid = chunk_mask[:, chunk_index].float().unsqueeze(-1)
            state, aux = self.field.step(u_j, state, valid=valid)
            last_z = aux["z_t"]
            last_c = state["c"]
            last_error = aux["error"]
            last_gate = aux["gate"]
            if return_state:
                codes.append(aux["a_t"])
                gates.append(aux["gate"])

        final_feat = torch.cat(
            [last_z, last_c, surface_mean, surface_max, last_error, last_gate],
            dim=-1,
        )
        logits = self.classifier(final_feat)
        if not return_state:
            return logits
        return {
            "logits": logits,
            "final_z": last_z,
            "final_c": last_c,
            "dictionary": state["D"],
            "codes": torch.stack(codes, dim=1) if codes else None,
            "gates": torch.stack(gates, dim=1) if gates else None,
            "chunk_mask": chunk_mask,
        }
