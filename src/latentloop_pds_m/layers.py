from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Pairwise even/odd RoPE rotation: [x0,x1,x2,x3] -> [-x1,x0,-x3,x2]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even for pairwise rotation")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Must match _rotate_half's even/odd layout. cat(freqs, freqs) is for split-half layouts.
        emb = freqs.repeat_interleave(2, dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, Dh], cos/sin: [L, Dh]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class GQACausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.q_proj = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(l, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        repeat = self.n_q_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        if attention_mask is None:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # PyTorch SDPA bool mask uses True = allowed. Keep causal + valid-key positions.
            key_allowed = attention_mask[:, None, None, :].to(torch.bool)
            causal_allowed = torch.ones((l, l), device=x.device, dtype=torch.bool).tril()[None, None, :, :]
            allow_mask = causal_allowed & key_allowed
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=allow_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        attn = attn.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.o_proj(attn)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        context_dim = context_dim or dim
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        b, l, _ = x.shape
        s = context.shape[1]
        q = self.q(x).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if context_mask is not None:
            # PyTorch SDPA bool mask uses True = allowed.
            attn_mask = context_mask[:, None, None, :].to(torch.bool)
        if causal:
            causal_allowed = torch.ones((l, s), device=x.device, dtype=torch.bool).tril()
            causal_allowed = causal_allowed[None, None, :, :]
            attn_mask = causal_allowed if attn_mask is None else (attn_mask & causal_allowed)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, l, self.dim)
        return self.o(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, ffn_dim: int, max_seq_len: int, rope_theta: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQACausalSelfAttention(d_model, n_q_heads, n_kv_heads, max_seq_len, rope_theta, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb.to(dtype=next(self.parameters()).dtype))


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def binary_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def pairwise_cosine_offdiag_loss(x: torch.Tensor) -> torch.Tensor:
    # x: [B, S, D]
    if x.shape[1] <= 1:
        return x.new_zeros(())
    x = F.normalize(x, dim=-1)
    sim = torch.matmul(x, x.transpose(-1, -2))
    s = sim.shape[-1]
    eye = torch.eye(s, device=x.device, dtype=torch.bool)[None]
    return sim.masked_select(~eye).abs().mean()


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return torch.tensor(0.0)
    return torch.stack([v if v.dim() else v[None] for v in values]).mean()
