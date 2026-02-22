"""Model components for mHC: RMSNorm, log-space Sinkhorn, connection modules, Transformer."""

import math
from typing import Dict, Any

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Implements Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed


class SinkhornLog(nn.Module):
    """Projects logits to a doubly stochastic matrix using log-space Sinkhorn-Knopp.

    Operating entirely in log-space avoids the exp/divide cycles of standard
    Sinkhorn, preventing overflow/underflow and giving more stable gradients.
    Temperature tau controls sharpness: smaller tau -> closer to a permutation.
    Matches the reference implementation's sinkhorn_log (arXiv:2409.19606).
    """

    def __init__(self, iterations: int = 10, tau: float = 0.05):
        super().__init__()
        self.iterations = iterations
        self.tau = tau

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        n = logits.shape[-1]
        Z = logits / self.tau
        log_marginal = torch.zeros(n, device=logits.device, dtype=logits.dtype)

        u = torch.zeros(logits.shape[:-1], device=logits.device, dtype=logits.dtype)
        v = torch.zeros_like(u)

        for _ in range(self.iterations):
            u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
            v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

        return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class ConnectionModule(nn.Module):
    """Base class for connection modules."""

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        raise NotImplementedError


class mHCModule(ConnectionModule):
    """Implements the Manifold-Constrained Hyper-Connection (mHC).
    Follows Equations (7) and (8) from the paper (arXiv:2512.24880).
    H_pre = sigmoid(unconstrained), H_post = 2*sigmoid(unconstrained),
    H_res = SinkhornLog(logits, tau) -> doubly stochastic.
    Uses log-space Sinkhorn for numerical stability (no exp/divide overflow).
    b_res is initialized to scaled identity so H_res starts near identity.
    """

    def __init__(self, C: int, n: int, sk_iters: int = 10, sk_tau: float = 0.05):
        super().__init__()
        self.n = n
        self.C = C
        self.norm = RMSNorm(n * C)

        self.phi_pre = nn.Linear(n * C, n, bias=False)
        self.phi_post = nn.Linear(n * C, n, bias=False)
        self.phi_res = nn.Linear(n * C, n * n, bias=False)

        self.a_pre = nn.Parameter(torch.full((1,), 0.01))
        self.b_pre = nn.Parameter(torch.zeros(1, 1, n))
        self.a_post = nn.Parameter(torch.full((1,), 0.01))
        self.b_post = nn.Parameter(torch.zeros(1, 1, n))
        self.a_res = nn.Parameter(torch.full((1,), 0.01))
        # Initialize b_res so diagonal logits dominate -> SinkhornLog ≈ I at init
        self.b_res = nn.Parameter(
            torch.eye(n).unsqueeze(0).unsqueeze(0) * 5.0
        )

        self.sinkhorn = SinkhornLog(iterations=sk_iters, tau=sk_tau)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        B, S, _, _ = x.shape

        x_flat = x.view(B, S, self.n * self.C)
        x_normed = self.norm(x_flat)

        H_pre_unconstrained = (
            self.a_pre * self.phi_pre(x_normed).unsqueeze(-2) + self.b_pre
        )
        H_post_unconstrained = (
            self.a_post * self.phi_post(x_normed).unsqueeze(-2) + self.b_post
        )
        H_res_unconstrained = (
            self.a_res * self.phi_res(x_normed).view(B, S, self.n, self.n) + self.b_res
        )

        H_pre = torch.sigmoid(H_pre_unconstrained)
        H_post = 2 * torch.sigmoid(H_post_unconstrained)
        H_res = self.sinkhorn(H_res_unconstrained)

        x_normed_reshaped = x_normed.view(B, S, self.n, self.C)
        h_in = torch.einsum("bsin,bsnc->bsc", H_pre, x_normed_reshaped)

        h_out = sublayer(h_in)

        h_post = torch.einsum(
            "bsni,bsc->bsnc", H_post.transpose(-1, -2), h_out
        )

        x_res = torch.einsum("bsij,bsjc->bsic", H_res, x)
        return x_res + h_post


class HCModule(ConnectionModule):
    """Implements the unconstrained Hyper-Connection (HC). Follows Equation (5)."""

    def __init__(self, C: int, n: int):
        super().__init__()
        self.n = n
        self.C = C
        self.norm = RMSNorm(n * C)

        self.phi_pre = nn.Linear(n * C, n, bias=False)
        self.phi_post = nn.Linear(n * C, n, bias=False)
        self.phi_res = nn.Linear(n * C, n * n, bias=False)

        self.a_pre = nn.Parameter(torch.full((1,), 0.01))
        self.b_pre = nn.Parameter(torch.zeros(1, 1, n))
        self.a_post = nn.Parameter(torch.full((1,), 0.01))
        self.b_post = nn.Parameter(torch.zeros(1, 1, n))
        self.a_res = nn.Parameter(torch.full((1,), 0.01))
        self.b_res = nn.Parameter(torch.zeros(1, 1, n, n))

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        B, S, _, _ = x.shape
        x_flat = x.view(B, S, self.n * self.C)
        x_normed = self.norm(x_flat)

        H_pre = torch.tanh(
            self.a_pre * self.phi_pre(x_normed).unsqueeze(-2) + self.b_pre
        )
        H_post = torch.tanh(
            self.a_post * self.phi_post(x_normed).unsqueeze(-2) + self.b_post
        )
        H_res = torch.tanh(
            self.a_res * self.phi_res(x_normed).view(B, S, self.n, self.n)
            + self.b_res
        )

        x_normed_reshaped = x_normed.view(B, S, self.n, self.C)
        h_in = torch.einsum("bsin,bsnc->bsc", H_pre, x_normed_reshaped)
        h_out = sublayer(h_in)
        h_post = torch.einsum(
            "bsni,bsc->bsnc", H_post.transpose(-1, -2), h_out
        )
        x_res = torch.einsum("bsij,bsjc->bsic", H_res, x)
        return x_res + h_post


class StandardConnection(ConnectionModule):
    """Implements the standard residual connection x + F(x)."""

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(x))


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with a pluggable connection module."""

    def __init__(self, C: int, n_heads: int, connection_module: ConnectionModule):
        super().__init__()
        self.norm1 = RMSNorm(C)
        self.attn = nn.MultiheadAttention(C, n_heads, batch_first=True)
        self.connection1 = connection_module

        self.norm2 = RMSNorm(C)
        ff_inner_dim = 4 * C
        self.ffn = nn.Sequential(
            nn.Linear(C, ff_inner_dim),
            nn.GELU(),
            nn.Linear(ff_inner_dim, C),
        )
        self.connection2 = connection_module

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_sublayer = lambda y: self.attn(
            y, y, y, attn_mask=mask, need_weights=False
        )[0]
        x = self.connection1(x, lambda y: attn_sublayer(self.norm1(y)))
        x = self.connection2(x, lambda y: self.ffn(self.norm2(y)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, C: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))
        pe = torch.zeros(1, max_len, C)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HyperTransformer(nn.Module):
    """Decoder-only Transformer with selectable connection type."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["C"])
        self.pos_encoder = PositionalEncoding(config["C"], config["max_seq_len"])
        self.dropout = nn.Dropout(config["dropout"])

        self.is_hyper = config["connection_type"] in ["hc", "mhc"]
        if self.is_hyper:
            self.initial_proj = nn.Linear(config["C"], config["n"] * config["C"])
            self.final_proj = nn.Linear(config["n"] * config["C"], config["C"])

        layers = []
        for _ in range(config["n_layers"]):
            if config["connection_type"] == "mhc":
                conn1 = mHCModule(
                    config["C"],
                    config["n"],
                    sk_iters=config.get("sinkhorn_iters", 10),
                    sk_tau=config.get("sinkhorn_tau", 0.05),
                )
                conn2 = mHCModule(
                    config["C"],
                    config["n"],
                    sk_iters=config.get("sinkhorn_iters", 10),
                    sk_tau=config.get("sinkhorn_tau", 0.05),
                )
            elif config["connection_type"] == "hc":
                conn1 = HCModule(config["C"], config["n"])
                conn2 = HCModule(config["C"], config["n"])
            else:
                conn1 = StandardConnection(config["dropout"])
                conn2 = conn1

            block = TransformerBlock(
                config["C"], config["n_heads"], conn1 if not self.is_hyper else None
            )
            if self.is_hyper:
                block.connection1 = conn1
                block.connection2 = conn2
            layers.append(block)

        self.layers = nn.ModuleList(layers)
        final_norm_dim = config["C"] * config["n"] if self.is_hyper else config["C"]
        self.final_norm = RMSNorm(final_norm_dim)
        self.lm_head = nn.Linear(config["C"], config["vocab_size"])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        mask = nn.Transformer.generate_square_subsequent_mask(S).to(input_ids.device)

        if self.is_hyper:
            x = self.initial_proj(x).view(
                B, S, self.config["n"], self.config["C"]
            )

            for layer in self.layers:
                attn_sublayer = lambda h_in: layer.attn(
                    h_in, h_in, h_in, attn_mask=mask, need_weights=False
                )[0]
                x = layer.connection1(x, attn_sublayer)

                ffn_sublayer = layer.ffn
                x = layer.connection2(x, ffn_sublayer)

            x = self.final_norm(x.view(B, S, -1))
            x = self.final_proj(x)
        else:
            for layer in self.layers:
                x = layer(x, mask)
            x = self.final_norm(x)

        logits = self.lm_head(x)
        return logits


def build_model_config(
    vocab_size: int,
    max_seq_len: int,
    connection_type: str = "mhc",
    C: int = 64,
    n: int = 4,
    n_layers: int = 12,
    n_heads: int = 4,
    dropout: float = 0.1,
    sinkhorn_iters: int = 10,
    sinkhorn_tau: float = 0.05,
) -> Dict[str, Any]:
    """Build model configuration dict."""
    return {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "C": C,
        "n": n,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "connection_type": connection_type,
        "sinkhorn_iters": sinkhorn_iters,
        "sinkhorn_tau": sinkhorn_tau,
    }
