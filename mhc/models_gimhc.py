"""GI-mHC: Gated Injection Hyper-Connections.

Extends mHC by splitting residual streams into n_tok token streams and n_ext
persistent memory streams. After each mHC forward pass, memory streams are
refreshed toward a learnable prior via a per-stream learned gate:

    x_ext_new = (1 - g) * x_ext_evolved + g * memory_init

The full (n_tok + n_ext)^2 H_res is doubly stochastic — the manifold
constraint guarantees memory streams always participate in mixing.
"""

import math
from typing import Dict, Any

import torch
import torch.nn as nn

from mhc.models import RMSNorm, SinkhornLog, ConnectionModule


class GIHCModule(ConnectionModule):
    """Gated Injection Hyper-Connection module.

    Runs a full mHC forward over n = n_tok + n_ext streams, then refreshes
    the last n_ext streams back toward a learnable memory prior using a
    per-stream sigmoid gate.

    Gate is initialized to sigmoid(-3) ≈ 0.05 so early training behaves
    almost identically to standard mHC. The model then learns how much to
    lean on persistent memory.
    """

    def __init__(
        self,
        C: int,
        n_tok: int,
        n_ext: int,
        sk_iters: int = 10,
        sk_tau: float = 0.05,
    ):
        super().__init__()
        self.C = C
        self.n_tok = n_tok
        self.n_ext = n_ext
        n = n_tok + n_ext
        self.n = n

        self.norm = RMSNorm(n * C)

        # mHC projection matrices — sized for full n streams
        self.phi_pre = nn.Linear(n * C, n, bias=False)
        self.phi_post = nn.Linear(n * C, n, bias=False)
        self.phi_res = nn.Linear(n * C, n * n, bias=False)

        # Learnable scale/bias for each connection matrix
        self.a_pre = nn.Parameter(torch.full((1,), 0.01))
        self.b_pre = nn.Parameter(torch.zeros(1, 1, n))
        self.a_post = nn.Parameter(torch.full((1,), 0.01))
        self.b_post = nn.Parameter(torch.zeros(1, 1, n))
        self.a_res = nn.Parameter(torch.full((1,), 0.01))
        # 5 * I_n so SinkhornLog starts near identity
        self.b_res = nn.Parameter(torch.eye(n).unsqueeze(0).unsqueeze(0) * 5.0)

        self.sinkhorn = SinkhornLog(iterations=sk_iters, tau=sk_tau)

        # GI-mHC extras
        self.memory_init = nn.Parameter(torch.randn(n_ext, C) * 0.02)
        # sigmoid(-3) ≈ 0.05 — almost no refresh at init
        self.gate_logit = nn.Parameter(torch.full((n_ext,), -3.0))

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
            x: (B, S, n_tok + n_ext, C)
            sublayer: callable (B, S, C) -> (B, S, C)
        Returns:
            (B, S, n_tok + n_ext, C)
        """
        B, S, _, _ = x.shape

        # --- Standard mHC forward over all n streams ---
        x_flat = x.view(B, S, self.n * self.C)
        x_normed = self.norm(x_flat)

        H_pre_unc = self.a_pre * self.phi_pre(x_normed).unsqueeze(-2) + self.b_pre
        H_post_unc = self.a_post * self.phi_post(x_normed).unsqueeze(-2) + self.b_post
        H_res_unc = (
            self.a_res * self.phi_res(x_normed).view(B, S, self.n, self.n)
            + self.b_res
        )

        H_pre = torch.sigmoid(H_pre_unc)
        H_post = 2 * torch.sigmoid(H_post_unc)
        H_res = self.sinkhorn(H_res_unc)

        x_normed_reshaped = x_normed.view(B, S, self.n, self.C)

        # Width connection: weighted combination of all streams → branch input
        h_in = torch.einsum("bsin,bsnc->bsc", H_pre, x_normed_reshaped)

        # Sublayer processes the combined single-stream input
        h_out = sublayer(h_in)

        # Depth connection: distribute branch output back to all streams
        h_post = torch.einsum("bsni,bsc->bsnc", H_post.transpose(-1, -2), h_out)

        # Residual mixing across streams (doubly stochastic)
        x_res = torch.einsum("bsij,bsjc->bsic", H_res, x)

        x_out = x_res + h_post

        # --- Gated injection: refresh memory streams toward learned prior ---
        g = torch.sigmoid(self.gate_logit)              # (n_ext,)
        g = g.view(1, 1, self.n_ext, 1)                # broadcast over B, S, C
        mem = self.memory_init.view(1, 1, self.n_ext, self.C)

        x_out_tok = x_out[:, :, : self.n_tok, :]
        x_out_ext = x_out[:, :, self.n_tok :, :]
        x_out_ext = (1.0 - g) * x_out_ext + g * mem

        return torch.cat([x_out_tok, x_out_ext], dim=2)


class GIHyperTransformer(nn.Module):
    """Decoder-only Transformer using GI-mHC connection modules.

    Token streams and memory streams are concatenated at layer 0.
    Only token streams are projected to the vocabulary at the end.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        C = config["C"]
        n_tok = config["n_tok"]
        n_ext = config["n_ext"]
        n = n_tok + n_ext
        self.n_tok = n_tok
        self.n_ext = n_ext
        self.n = n

        self.embedding = nn.Embedding(config["vocab_size"], C)
        self.pos_encoder = _PositionalEncoding(C, config["max_seq_len"])
        self.dropout = nn.Dropout(config["dropout"])

        # Project token embedding into n_tok streams
        self.initial_proj = nn.Linear(C, n_tok * C)
        # Project n_tok token streams back to C for LM head
        self.final_proj = nn.Linear(n_tok * C, C)

        sk_iters = config.get("sinkhorn_iters", 10)
        sk_tau = config.get("sinkhorn_tau", 0.05)

        self.layers = nn.ModuleList()
        for _ in range(config["n_layers"]):
            layer = _GITransformerBlock(C, config["n_heads"], n_tok, n_ext, sk_iters, sk_tau)
            self.layers.append(layer)

        self.final_norm = RMSNorm(n_tok * C)
        self.lm_head = nn.Linear(C, config["vocab_size"])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape

        x = self.embedding(input_ids)       # (B, S, C)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Token streams: (B, S, n_tok, C)
        x_tok = self.initial_proj(x).view(B, S, self.n_tok, self.C_dim)

        # Memory streams: tile memory_init from layer 0 across (B, S)
        mem0 = self.layers[0].conn1.memory_init  # (n_ext, C)
        x_ext = mem0.view(1, 1, self.n_ext, self.C_dim).expand(B, S, -1, -1)

        # Full state: (B, S, n_tok + n_ext, C)
        x = torch.cat([x_tok, x_ext], dim=2)

        mask = nn.Transformer.generate_square_subsequent_mask(S).to(input_ids.device)

        for layer in self.layers:
            x = layer(x, mask)

        # Read only token streams for output
        x_tok_out = x[:, :, : self.n_tok, :].contiguous().view(B, S, self.n_tok * self.C_dim)
        x_tok_out = self.final_norm(x_tok_out)
        x_tok_out = self.final_proj(x_tok_out)

        return self.lm_head(x_tok_out)

    @property
    def C_dim(self) -> int:
        return self.config["C"]


class _GITransformerBlock(nn.Module):
    """Transformer block (attn + FFN) using two GIHCModule connections."""

    def __init__(
        self,
        C: int,
        n_heads: int,
        n_tok: int,
        n_ext: int,
        sk_iters: int,
        sk_tau: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(C)
        self.attn = nn.MultiheadAttention(C, n_heads, batch_first=True)
        self.conn1 = GIHCModule(C, n_tok, n_ext, sk_iters, sk_tau)

        self.norm2 = RMSNorm(C)
        ff_inner = 4 * C
        self.ffn = nn.Sequential(
            nn.Linear(C, ff_inner),
            nn.GELU(),
            nn.Linear(ff_inner, C),
        )
        self.conn2 = GIHCModule(C, n_tok, n_ext, sk_iters, sk_tau)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def attn_sublayer(h):
            return self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]

        x = self.conn1(x, lambda h: attn_sublayer(self.norm1(h)))
        x = self.conn2(x, lambda h: self.ffn(self.norm2(h)))
        return x


class _PositionalEncoding(nn.Module):
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


def build_gimhc_config(
    vocab_size: int,
    max_seq_len: int,
    C: int = 64,
    n_tok: int = 4,
    n_ext: int = 1,
    n_layers: int = 12,
    n_heads: int = 4,
    dropout: float = 0.1,
    sinkhorn_iters: int = 10,
    sinkhorn_tau: float = 0.05,
) -> Dict[str, Any]:
    """Build config dict for GIHyperTransformer."""
    return {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "C": C,
        "n_tok": n_tok,
        "n_ext": n_ext,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "sinkhorn_iters": sinkhorn_iters,
        "sinkhorn_tau": sinkhorn_tau,
    }
