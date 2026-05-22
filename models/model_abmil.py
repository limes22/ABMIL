"""
ABMIL (Attention-Based MIL, Ilse et al. 2018) implementation,
extended with optional ECSA plug-in (§4.2 of the paper).

Use:
    # Vanilla ABMIL (soft aggregation)
    model = ABMIL(embed_dim=512, n_classes=2)

    # ABMIL + ECSA (sparse aggregation with entropy-driven k)
    model = ABMIL(embed_dim=512, n_classes=2, use_ecsa=True,
                  ecsa_kwargs={'c_min': 4, 'c_max': 32,
                               'k_min_pct': 0.001, 'k_max_pct': 0.01,
                               'gamma': 1.0, 'inverse': False})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_clam import (Attn_Net_Gated, Attn_Net, FeatureAnalyzer,
                                AdaptiveSparsePooling, AdaptiveNormalization,
                                AdaptiveActivation, AdaptiveTemperature,
                                LSAPTemperature)
from models.ecsa import ECSA


class ABMIL(nn.Module):
    """
    Attention-Based MIL (Ilse et al., ICML 2018).

    Bag embedding:  z = Σ_n a_n h_n  (soft aggregation)
    or, with ECSA:  z = Σ_{n ∈ T_i} ã_n h_n  (sparse aggregation, see §4.2)

    No instance-level loss (in contrast to CLAM).
    """

    def __init__(
        self,
        size_arg='small',
        embed_dim=1024,
        n_classes=2,
        dropout=0.25,
        gate=True,
        use_ecsa=False,
        ecsa_kwargs=None,
        adaptive_sparse_pool=False,
        feature_adaptive=False,
        k_max_cap=0,
        dk_inverse=False,
        entropy_only=False,
        entropy_k_floor=8,
        loglinear=False,
        loglinear_k_min=8,
        loglinear_k_cap=500,
        loglinear_cap_frac=0.0,
        k_min_pct=0.001,
        k_max_pct=0.01,
        gamma=1.0,
        entropy_method='v2',
        inverse_threshold=1.0,
        hybrid_floor=False,
        hybrid_floor_alpha=0.0,
        hybrid_floor_min=8,
        blend_w_bias=0.0,
        pure_cap=False,
        pure_cap_frac=0.0,
        pure_cap_min=8,
        learnable_alpha=False,
        learnable_alpha_init=0.03,
        learnable_alpha_temp=20.0,
        learnable_alpha_min=8,
        learnable_alpha_hybrid=False,
        attn_norm='softmax',
        lsap_temp=False,
        lsap_eps=0.01,
        lsap_no_tau=False,   # True: 표준화 유지, τ=1 고정 (τ-MLP 제거)
        lsap_alpha=1.5,      # entmax α 값 (1.1, 1.3, 1.5, 1.7, ...) entmax_bisect 사용
    ):
        super().__init__()
        size_dict = {
            'small': [embed_dim, 512, 256],
            'big':   [embed_dim, 512, 384],
        }
        size = size_dict[size_arg]

        # Phase 2: feature_adaptive 메타-flag 모든 5 모듈 활성
        self.feature_adaptive = feature_adaptive
        # adaptive_sparse_pool 자동 활성화 (featad 의 sub-module)
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
        # Attention normalization. 'softmax' | 'sparsemax' | 'entmax15' | 'entmax_alpha'
        # 'entmax_alpha' → entmax_bisect with arbitrary α (lsap_alpha).
        self.attn_norm = attn_norm
        self.lsap_temp = lsap_temp
        self.lsap_no_tau = lsap_no_tau
        self.lsap_alpha = lsap_alpha
        # AdaptiveNormalization / AdaptiveActivation 은 feature_adaptive 일 때만 부속.
        if feature_adaptive:
            self.adaptive_norm = AdaptiveNormalization(embed_dim)
            self.adaptive_act = AdaptiveActivation()
        # Temperature module 또는 standardization-only (τ=1) 활성 조건:
        #   feature_adaptive=True or lsap_temp=True → learned τ
        #   lsap_no_tau=True → τ=1 고정 (표준화 path 만 살림)
        if feature_adaptive or lsap_temp:
            if lsap_temp:
                self.temp_module = LSAPTemperature(eps=lsap_eps)
            else:
                self.temp_module = AdaptiveTemperature(n_attn_heads=1)
        elif lsap_no_tau:
            # Marker only — forward 에서 표준화 path 활성. temp_module 은 None.
            self.temp_module = None
        else:
            self.temp_module = None

        # featad 모드에서는 AdaptiveActivation 이 처리하므로 ReLU → Identity
        activation = nn.Identity() if feature_adaptive else nn.ReLU()
        fc = [nn.Linear(size[0], size[1]), activation, nn.Dropout(dropout)]
        if gate:
            attn = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attn = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attn)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        self.use_ecsa = use_ecsa
        if use_ecsa:
            self.ecsa = ECSA(**(ecsa_kwargs or {}))
        else:
            self.ecsa = None

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        """
        Returns (logits, Y_prob, Y_hat, A_raw, results_dict) for compatibility with CLAM trainer.
        """
        # Phase 2: feature-adaptive normalization (입력)
        stats_vec = self.feat_analyzer(h) if self.adaptive_sparse_pool else None
        if self.feature_adaptive:
            h = self.adaptive_norm(h, stats_vec)

        A, h_red = self.attention_net(h)            # A: [N, 1], h_red: [N, 512]

        # Phase 2: feature-adaptive activation (attention_net 출력)
        if self.feature_adaptive:
            h_red = self.adaptive_act(h_red, stats_vec)

        A = torch.transpose(A, 1, 0)                # [1, N]
        if attention_only:
            return A
        A_raw = A

        # Temperature/standardization path. Three modes:
        #   (1) temp_module != None : feature_adaptive or lsap_temp — z-score + τ
        #   (2) lsap_no_tau         : z-score only, τ=1 fixed
        #   (3) else                : no scaling
        if self.temp_module is not None:
            temp = self.temp_module(A_raw)
            if self.lsap_temp:
                mu = A.mean(); sigma = A.std()
                if not torch.isfinite(sigma) or sigma < 1e-8:
                    sigma = torch.tensor(1.0, device=A.device, dtype=A.dtype)
                A_scaled = (A - mu) / sigma / temp
            else:
                A_scaled = A / temp
        elif self.lsap_no_tau:
            # τ=1 고정, 표준화만. LSAP ablation: "τ-MLP 부수적" 검증.
            mu = A.mean(); sigma = A.std()
            if not torch.isfinite(sigma) or sigma < 1e-8:
                sigma = torch.tensor(1.0, device=A.device, dtype=A.dtype)
            A_scaled = (A - mu) / sigma
        else:
            A_scaled = A

        # Attention normalization: softmax | sparsemax | entmax15 | entmax_alpha
        if self.attn_norm == 'sparsemax':
            from entmax import sparsemax
            A_softmax = sparsemax(A_scaled, dim=1)
        elif self.attn_norm == 'entmax15':
            from entmax import entmax15
            A_softmax = entmax15(A_scaled, dim=1)
        elif self.attn_norm == 'entmax_alpha':
            from entmax import entmax_bisect
            A_softmax = entmax_bisect(A_scaled, alpha=self.lsap_alpha, dim=1)
        else:
            A_softmax = F.softmax(A_scaled, dim=1)  # [1, N]

        bag_size_N = A_softmax.shape[1]
        sparse_ratio = None
        ecsa_info = {}

        if self.adaptive_sparse_pool:
            z, k_pool, sparse_ratio = self.adaptive_pool(A_softmax, h_red, stats_vec)
        elif self.use_ecsa:
            z, ecsa_info = self.ecsa(A_softmax, h_red)
            k_pool = ecsa_info.get('k', bag_size_N)
        else:
            z = torch.mm(A_softmax, h_red)          # [1, 512]
            # For LSAP/entmax/sparsemax: k_pool = ||p||_0 (nonzero count) — meaningful k.
            # For softmax: k_pool = N (all patches contribute).
            if self.attn_norm in ('entmax15', 'sparsemax', 'entmax_alpha'):
                k_pool = int((A_softmax > 1e-6).sum().item())
            else:
                k_pool = bag_size_N

        logits = self.classifier(z)                 # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=1, keepdim=True)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {
            'ecsa': ecsa_info,
            'num_patches': bag_size_N,
            'k_pool': k_pool,
            # CLAM trainer 가 instance_loss 를 기대하므로 placeholder
            'instance_loss': torch.tensor(0.0, device=h.device),
            'inst_labels': [], 'inst_preds': [],
        }
        if sparse_ratio is not None:
            results_dict['sparse_ratio'] = sparse_ratio
        if return_features:
            results_dict['features'] = z

        return logits, Y_prob, Y_hat, A_raw, results_dict
