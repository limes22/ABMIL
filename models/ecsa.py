"""
ECSA (Entropy-Conditional Sparse Aggregation) — host-agnostic plug-in module.

Use case:
    >>> from models.ecsa import ECSA
    >>> ecsa = ECSA(c_min=4, c_max=32, k_min_pct=0.001, k_max_pct=0.01, gamma=1.0)
    >>> # A_softmax: [1, N] or [N], softmax-normalized attention
    >>> # h:         [N, D], instance embeddings
    >>> z, info = ecsa(A_softmax, h)  # z: [1, D] sparse bag embedding

ECSA works with ANY attention-based MIL model whose forward pass produces
(softmax_attention, instance_embeddings). Plug it in by replacing the standard
soft aggregation z = Σ a_n h_n with z = ECSA(a, h).

Reference: see §4.2 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dynamic_k_from_entropy(
    A_softmax: torch.Tensor,
    c_min: int = 4,
    c_max: int = 32,
    k_min_pct: float = 0.001,
    k_max_pct: float = 0.01,
    gamma: float = 1.0,
    entropy_method: str = 'v1',
    inverse: bool = False,
):
    """
    Compute per-slide cardinality k_i from softmax-normalized attention.

    Args:
        A_softmax: [1, N] or [N]. Must already be softmax-normalized.
        c_min, c_max: hard floors for k_min and k_max bounds.
        k_min_pct, k_max_pct: proportional bounds (k_min = max(c_min, N * k_min_pct)).
        gamma: curvature of entropy → k mapping.
        entropy_method: 'v1' (full Shannon entropy) or 'v2' (top-M + bot-M truncated).
        inverse: if True, sharp attention → large k (default False; see §4.2.3).

    Returns:
        k: int, cardinality in [k_min, k_max].
        H_norm: float, normalized entropy ∈ [0, 1].
        H_raw: float, raw entropy.
    """
    if A_softmax.dim() == 1:
        A_softmax = A_softmax.unsqueeze(0)

    N = A_softmax.shape[1]
    eps = 1e-8

    # Adaptive bounds: max(hard_floor, N * pct)
    k_min = max(c_min, int(N * k_min_pct))
    k_max = max(c_max, int(N * k_max_pct))
    if k_max < k_min:
        k_max = k_min

    # Entropy estimator
    if entropy_method == 'v2':
        half_M = min(250, N // 2)
        if half_M < 1:
            return max(k_min, 1), 0.0, 0.0
        top_vals = torch.topk(A_softmax, half_M, dim=1)[0]
        bot_vals = torch.topk(A_softmax, half_M, dim=1, largest=False)[0]
        sel = torch.cat([top_vals, bot_vals], dim=1)
        sel = sel / (sel.sum(dim=1, keepdim=True) + eps)
        M_total = sel.shape[1]
        H = -torch.sum(sel * torch.log(sel + eps))
        H_max = torch.log(torch.tensor(float(M_total), device=A_softmax.device))
    else:
        H = -torch.sum(A_softmax * torch.log(A_softmax + eps))
        H_max = torch.log(torch.tensor(float(N), device=A_softmax.device))

    if torch.isnan(H):
        return k_min, 0.0, 0.0

    H_norm = (H / H_max).clamp(0.0, 1.0)

    # Forward (default) or inverse mapping
    if inverse:
        k = int(k_min + (k_max - k_min) * ((1.0 - H_norm.item()) ** gamma))
    else:
        k = int(k_min + (k_max - k_min) * (H_norm.item() ** gamma))

    # Safety: 2k <= N (for downstream instance loss compatibility)
    k = min(k, max(1, (N - 1) // 2))
    k = max(k, 1)

    return k, H_norm.item(), H.item()


class ECSA(nn.Module):
    """
    Entropy-Conditional Sparse Aggregation.

    A drop-in replacement for the standard MIL soft aggregation
        z = Σ_n a_n · h_n
    that adapts the per-slide aggregation cardinality k_i from the
    attention entropy and aggregates only over the top-k_i instances.

    Adds zero learnable parameters. Use as a forward-time module after the
    host model produces its softmax-normalized attention vector.
    """

    def __init__(
        self,
        c_min: int = 4,
        c_max: int = 32,
        k_min_pct: float = 0.001,
        k_max_pct: float = 0.01,
        gamma: float = 1.0,
        entropy_method: str = 'v1',
        inverse: bool = False,
    ):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        self.k_min_pct = k_min_pct
        self.k_max_pct = k_max_pct
        self.gamma = gamma
        self.entropy_method = entropy_method
        self.inverse = inverse

    def forward(self, A_softmax: torch.Tensor, h: torch.Tensor):
        """
        Args:
            A_softmax: [1, N] softmax-normalized attention (rows sum to 1).
            h:         [N, D] instance embeddings.

        Returns:
            z:    [1, D] sparse bag embedding.
            info: dict with selected k, entropy stats, sparse attention indices.
        """
        if A_softmax.dim() == 1:
            A_softmax = A_softmax.unsqueeze(0)

        # Step 1: cardinality from entropy
        k_i, H_norm, H_raw = compute_dynamic_k_from_entropy(
            A_softmax,
            c_min=self.c_min, c_max=self.c_max,
            k_min_pct=self.k_min_pct, k_max_pct=self.k_max_pct,
            gamma=self.gamma, entropy_method=self.entropy_method,
            inverse=self.inverse,
        )

        # Step 2: top-k indices and renormalization
        N = A_softmax.shape[1]
        k_pool = min(k_i, N)
        top_vals, top_idx = torch.topk(A_softmax, k_pool, dim=1)  # [1, k_pool]
        A_sparse = torch.zeros_like(A_softmax)
        A_sparse.scatter_(1, top_idx, top_vals)
        A_sparse = A_sparse / (A_sparse.sum(dim=1, keepdim=True) + 1e-8)

        # Step 3: sparse aggregation
        z = torch.mm(A_sparse, h)  # [1, D]

        info = {
            'k': k_i,
            'entropy_norm': H_norm,
            'entropy_raw': H_raw,
            'top_idx': top_idx,
            'sparse_attention': A_sparse,
        }
        return z, info
