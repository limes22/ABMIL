"""
ACMIL (Zhang et al., ECCV 2024) wrapper with optional ECSA plug-in.

Adapts ACMIL/architecture/transformer.py's ACMIL_GA to our k-fold CV pipeline
and adds an ECSA option that replaces the slide-level soft aggregation.

References:
  - ACMIL paper: arXiv:2311.07125
  - Original code: https://github.com/dazhangyu123/ACMIL
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ACMIL repo path 등록
ACMIL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ACMIL')
if ACMIL_PATH not in sys.path:
    sys.path.insert(0, ACMIL_PATH)

from architecture.transformer import Attention_Gated  # type: ignore
from architecture.network import Classifier_1fc, DimReduction  # type: ignore

from models.ecsa import ECSA
from models.model_clam import (FeatureAnalyzer, AdaptiveSparsePooling,
                                AdaptiveNormalization, AdaptiveActivation,
                                AdaptiveTemperature, LSAPTemperature)


class ACMILWithECSA(nn.Module):
    """
    ACMIL_GA architecture (Multi-Branch Attention + STKIM) with optional ECSA plug-in.

    When use_ecsa=False: matches the original ACMIL paper's architecture.
    When use_ecsa=True:  ECSA replaces the slide-level soft aggregation step
                         (bag_feat = bag_A @ x  →  z = ECSA(bag_A, x)).

    Per-branch aggregations (afeat = A_branch @ x) remain unchanged so that
    the multi-branch sub-classifiers and diversity loss continue to work.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        D_inner: int = 512,                  # bag-feature dim after DimReduction
        D_attn: int = 128,                   # attention hidden dim
        n_classes: int = 2,
        n_token: int = 5,                    # MBA branches
        n_masked_patch: int = 10,            # STKIM cardinality
        mask_drop: float = 0.6,              # STKIM drop probability
        droprate: float = 0.0,
        use_ecsa: bool = False,
        ecsa_kwargs=None,
        adaptive_sparse_pool: bool = False,
        feature_adaptive: bool = False,
        k_max_cap: int = 0,
        dk_inverse: bool = False,
        entropy_only: bool = False,
        entropy_k_floor: int = 8,
        loglinear: bool = False,
        loglinear_k_min: int = 8,
        loglinear_k_cap: int = 500,
        loglinear_cap_frac: float = 0.0,
        k_min_pct: float = 0.001,
        k_max_pct: float = 0.01,
        gamma: float = 1.0,
        entropy_method: str = 'v2',
        inverse_threshold: float = 1.0,
        hybrid_floor: bool = False,
        hybrid_floor_alpha: float = 0.0,
        hybrid_floor_min: int = 8,
        blend_w_bias: float = 0.0,
        pure_cap: bool = False,
        pure_cap_frac: float = 0.0,
        pure_cap_min: int = 8,
        learnable_alpha: bool = False,
        learnable_alpha_init: float = 0.03,
        learnable_alpha_temp: float = 20.0,
        learnable_alpha_min: int = 8,
        learnable_alpha_hybrid: bool = False,
        attn_norm: str = 'softmax',
        lsap_temp: bool = False,
        lsap_eps: float = 0.01,
    ):
        super().__init__()
        self.dimreduction = DimReduction(embed_dim, D_inner)
        self.attention = Attention_Gated(L=D_inner, D=D_attn, K=n_token)
        self.classifier = nn.ModuleList([
            Classifier_1fc(D_inner, n_classes, droprate) for _ in range(n_token)
        ])
        self.Slide_classifier = Classifier_1fc(D_inner, n_classes, droprate)
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.n_classes = n_classes

        self.use_ecsa = use_ecsa
        self.ecsa = ECSA(**(ecsa_kwargs or {})) if use_ecsa else None

        # Phase 1+2: adaptive modules
        self.feature_adaptive = feature_adaptive
        self.adaptive_sparse_pool = adaptive_sparse_pool or feature_adaptive
        if self.adaptive_sparse_pool:
            self.feat_analyzer = FeatureAnalyzer()
            self.adaptive_pool = AdaptiveSparsePooling(
                k_min_pct=k_min_pct, k_max_pct=k_max_pct, gamma=gamma, entropy_method=entropy_method,
                k_max_cap=k_max_cap, inverse=dk_inverse,
                entropy_only=entropy_only, entropy_k_floor=entropy_k_floor,
                loglinear=loglinear, loglinear_k_min=loglinear_k_min, loglinear_k_cap=loglinear_k_cap, loglinear_cap_frac=loglinear_cap_frac,
                inverse_threshold=inverse_threshold, hybrid_floor=hybrid_floor, hybrid_floor_alpha=hybrid_floor_alpha, hybrid_floor_min=hybrid_floor_min,
                blend_w_bias=blend_w_bias, pure_cap=pure_cap, pure_cap_frac=pure_cap_frac, pure_cap_min=pure_cap_min,
                learnable_alpha=learnable_alpha, learnable_alpha_init=learnable_alpha_init,
                learnable_alpha_temp=learnable_alpha_temp, learnable_alpha_min=learnable_alpha_min,
                learnable_alpha_hybrid=learnable_alpha_hybrid)
        # Attention normalization (softmax | sparsemax | entmax15) for slide-level pooling.
        self.attn_norm = attn_norm
        self.lsap_temp = lsap_temp
        if feature_adaptive:
            self.adaptive_norm = AdaptiveNormalization(embed_dim)
            self.adaptive_act = AdaptiveActivation()
            if lsap_temp:
                self.temp_module = LSAPTemperature(eps=lsap_eps)
            else:
                self.temp_module = AdaptiveTemperature(n_attn_heads=n_token)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        """
        Args:
            h: [N, embed_dim] frozen instance features.

        Returns:
            (slide_logits, Y_prob, Y_hat, A_raw, results_dict)
              - slide_logits: [1, n_classes] slide-level prediction
              - results_dict contains:
                  'sub_logits':  [n_token, n_classes] per-branch predictions
                  'attn_softmax': [n_token, N] softmax attention (for diff_loss)
                  'ecsa': dict with k_i, entropy_norm, etc.
        """
        # Phase 2: feature-adaptive normalization (입력 단계)
        stats_vec = self.feat_analyzer(h) if self.adaptive_sparse_pool else None
        if self.feature_adaptive:
            h = self.adaptive_norm(h, stats_vec)

        # ACMIL expects 3D input (B,N,L); we receive 2D (N,L).
        x = self.dimreduction(h)             # [N, D_inner]

        # Phase 2: feature-adaptive activation (DimReduction 후)
        if self.feature_adaptive:
            x = self.adaptive_act(x, stats_vec)

        A = self.attention(x)                # [n_token, N]

        # Phase 2: adaptive temperature (per-slide)
        if self.feature_adaptive:
            A = A / self.temp_module(A)

        # STKIM masking (training-time only)
        if self.n_masked_patch > 0 and self.training:
            k_branches, n = A.shape
            n_mask = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_mask, dim=-1)
            rand_sel = torch.argsort(torch.rand(*indices.shape, device=A.device), dim=-1)[:, :int(n_mask * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_sel]
            mask = torch.ones(k_branches, n, device=A.device)
            mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(mask == 0, -1e9)

        A_out = A                            # [n_token, N], pre-softmax
        if attention_only:
            return A_out

        # Attention normalization: softmax | sparsemax | entmax15. Per-head.
        if self.attn_norm == 'sparsemax':
            from entmax import sparsemax
            A_softmax = sparsemax(A, dim=1)
        elif self.attn_norm == 'entmax15':
            from entmax import entmax15
            A_softmax = entmax15(A, dim=1)
        else:
            A_softmax = F.softmax(A, dim=1)  # [n_token, N]

        # Per-branch bag features (kept for sub-classifier path)
        afeat = torch.mm(A_softmax, x)       # [n_token, D_inner]
        sub_logits = torch.stack([head(afeat[i]) for i, head in enumerate(self.classifier)], dim=0)
        # sub_logits: [n_token, n_classes]

        # Slide-level aggregation: average branch attentions, then aggregate
        bag_A = A_softmax.mean(0, keepdim=True)   # [1, N]
        bag_size_N = bag_A.shape[1]

        sparse_ratio = None
        ecsa_info = {}
        if self.adaptive_sparse_pool:
            # stats_vec 가 forward 초반에 계산됐으면 reuse
            if stats_vec is None:
                stats_vec = self.feat_analyzer(h)
            z, k_pool, sparse_ratio = self.adaptive_pool(bag_A, x, stats_vec)
        elif self.use_ecsa:
            z, ecsa_info = self.ecsa(bag_A, x)    # z: [1, D_inner]
            k_pool = ecsa_info.get('k', bag_size_N)
        else:
            z = torch.mm(bag_A, x)                # [1, D_inner]
            k_pool = bag_size_N

        slide_logits = self.Slide_classifier(z)    # [1, n_classes]
        Y_hat = torch.argmax(slide_logits, dim=1, keepdim=True)
        Y_prob = F.softmax(slide_logits, dim=1)

        results_dict = {
            'sub_logits': sub_logits,
            'attn_softmax': A_softmax,
            'ecsa': ecsa_info,
            'num_patches': bag_size_N,
            'k_pool': k_pool,
        }
        if sparse_ratio is not None:
            results_dict['sparse_ratio'] = sparse_ratio
        if return_features:
            results_dict['features'] = z
        return slide_logits, Y_prob, Y_hat, A_out, results_dict
