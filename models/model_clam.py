import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class AdaptiveTemperature(nn.Module):
    """
    Slide-adaptive attention temperature.

    Raw attention score의 통계(mean, std, max, min)를 입력으로 받아
    각 슬라이드에 최적화된 temperature를 예측하는 네트워크.

    명확한 슬라이드 → temp 낮게 → sharp attention → 소수 패치 집중
    애매한 슬라이드 → temp 높게 → flat attention → 넓게 탐색
    """
    def __init__(self, n_attn_heads=1):
        super().__init__()
        # 입력: [mean, std, max, min] of raw attention per head → 4 features
        self.temp_net = nn.Sequential(
            nn.Linear(4 * n_attn_heads, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # 초기화: temp = 1.0 (vanilla softmax 와 동일하게 시작) 하도록 bias 조정.
        # 이전 버전은 bias=0 으로 초기 temp=1.5 → CTransPath 등 weak feature 에서
        # 학습 초반 attention 이 너무 평탄해져 일부 fold 가 local minimum 에 빠지는
        # 학습 불안정성 발생. ln(1/3) ≈ -1.0986 로 sigmoid(.) = 0.25 → temp = 1.0.
        import math as _math
        nn.init.zeros_(self.temp_net[-1].weight)
        nn.init.constant_(self.temp_net[-1].bias, _math.log(1.0 / 3.0))

    def forward(self, A_raw):
        """
        Args:
            A_raw: raw attention logits [K, N] (before softmax)
        Returns:
            temperature: scalar (0.5 ~ 2.5 범위)
        """
        # 각 head별 통계 추출
        stats = []
        for k in range(A_raw.shape[0]):
            a = A_raw[k]
            stats.extend([a.mean(), a.std(), a.max(), a.min()])
        stats_tensor = torch.stack(stats).unsqueeze(0)  # [1, 4*K]

        # temperature 예측 (0.5 ~ 2.5 범위로 제한)
        raw_out = self.temp_net(stats_tensor)  # [1, 1]
        temp = 0.5 + 2.0 * torch.sigmoid(raw_out)  # [0.5, 2.5]
        return temp.squeeze()

    def get_temperature_from_raw(self, A_raw):
        """로깅용: temperature 값 반환"""
        with torch.no_grad():
            return self.forward(A_raw).item()


class LSAPTemperature(nn.Module):
    """LSAP τ-MLP (Learned Sparse Attention Pooling).

    Input: 4-dim slide summary s = [μ_a, σ_a, max(a), log N] of raw attention logits a.
           s is DETACHED — τ-MLP gradient does not flow back into attention logits.
    Output: τ = softplus(MLP_φ(s)) + ε    (unbounded above, floor ε)

    Pair with the standardization a~ = (a - μ_a)/σ_a before division by τ in the host
    model's forward — without that, the scale of a and τ are entangled (a×2 ≡ τ÷2)
    and τ cannot uniquely control sparsity.

    Sparse-pooling formula: p = entmax_1.5(a~ / τ).  Used with --attn_norm entmax15.
    τ small → very sparse;  τ large → close to softmax.
    """
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps
        # LayerNorm 으로 4-dim input scale 정렬: [μ_a, σ_a, max(a), log N] 의 자연 스케일이
        # ~1, ~1, ~3, ~7 로 7배 차이 → log N 이 출력을 지배하는 문제 (sanity v2 r(N,τ)=-0.951).
        # LayerNorm 후 모든 stats 가 동등한 발언권 가짐 → σ_a 등 attention 분포 신호가
        # τ 결정에 기여 가능.
        self.input_norm = nn.LayerNorm(4)
        self.tau_net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Init τ ≈ 1.0 (neutral start, matches standard softmax scale on a~).
        # weight: small normal (std=0.01) so τ-MLP has gradient signal from epoch 0.
        # zero init 은 weight gradient 가 0 에서 멈춰서 stats 가 변해도 출력 거의 동일 →
        # τ가 슬라이드별로 적응 안 함 (실측 std=0.0008, sanity v1 결과).
        # bias 는 그대로 softplus^-1(1-ε) 로 두어 초기 τ≈1.0 유지.
        import math as _math
        nn.init.normal_(self.tau_net[-1].weight, mean=0.0, std=0.01)
        target = max(1e-3, 1.0 - eps)
        bias_init = _math.log(_math.exp(target) - 1.0)   # softplus^-1(target)
        nn.init.constant_(self.tau_net[-1].bias, bias_init)

    def forward(self, A_raw):
        """A_raw: [K, N] raw logits (pre-softmax). For K>1 (ACMIL), mean across heads.

        s is computed from a.detach() — the τ branch is gradient-isolated from the
        attention branch. This implements [3] of the LSAP spec.
        """
        if A_raw.dim() == 2 and A_raw.shape[0] > 1:
            a = A_raw.mean(dim=0)            # [N]
        else:
            a = A_raw.reshape(-1)            # [N]
        a_det = a.detach()                   # ← gradient stops here
        N = a_det.shape[0]
        log_N = torch.log(torch.tensor(float(max(N, 1)), device=a.device, dtype=a.dtype))
        std = a_det.std()
        if not torch.isfinite(std):          # N=1 edge case
            std = torch.tensor(1.0, device=a.device, dtype=a.dtype)
        stats = torch.stack([a_det.mean(), std, a_det.max(), log_N]).unsqueeze(0)  # [1, 4]
        stats = self.input_norm(stats)       # ← 입력 스케일 정렬 (모든 stats 동등 발언권)
        tau_raw = self.tau_net(stats)        # [1, 1]
        tau = F.softplus(tau_raw) + self.eps
        return tau.squeeze()

    def get_temperature_from_raw(self, A_raw):
        with torch.no_grad():
            return self.forward(A_raw).item()


class AdaptiveNormalization(nn.Module):
    """Phase 2: feature 분포에 따라 normalization 방식 (LayerNorm / Identity / Std)
    의 비중을 gate 가 학습. Backbone-specific feature scale 차이를 자동 보정."""
    def __init__(self, D):
        super().__init__()
        self.layer_norm = nn.LayerNorm(D)
        self.gate_net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Softmax(dim=-1),
        )

    def forward(self, h, stats_vec):
        gate = self.gate_net(stats_vec)
        h1 = self.layer_norm(h)
        h2 = h
        h3 = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        return gate[0] * h1 + gate[1] * h2 + gate[2] * h3


class AdaptiveActivation(nn.Module):
    """Phase 2: ReLU / GELU / SiLU 의 비중을 feature 분포 따라 gate 가 학습.
    음수 많은 CTransPath → GELU 비중 ↑; 양수만인 ResNet → ReLU 비중 ↑."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
        self.gate_net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Softmax(dim=-1),
        )

    def forward(self, h, stats_vec):
        gate = self.gate_net(stats_vec)
        return gate[0] * self.relu(h) + gate[1] * self.gelu(h) + gate[2] * self.silu(h)


class FeatureAnalyzer(nn.Module):
    """
    Phase 1 (Feature_Adaptive 통합): patch feature 의 분포 통계를 추출.
    Backbone 마다 (CTrans / UNI / Virchow / CONCH / ResNet) feature 분포가 다른데,
    이 통계 vector 를 downstream gate 가 보고 최적 동작 학습.
    """
    def forward(self, h):
        # h: [N, D] patch features
        stats_vec = torch.stack([
            h.mean(),
            h.std(),
            h.min(),
            h.max(),
            (h < 0).float().mean(),                # 음수 비율 (CTrans 같은 zero-centered features)
            (h.abs() < 0.01).float().mean(),       # 희소성
        ])
        return stats_vec


class AdaptiveSparsePooling(nn.Module):
    """
    Dual-adaptive sparse aggregation (Phase 1 enhanced):
      • k_i  = entropy-driven  (DK 와 동일 로직, [N·k_min_pct, N·k_max_pct] 범위)
      • alpha = gate-driven    (feature stats → sparse vs dense blend 비중)

    M = alpha · M_sparse(top-k_i) + (1 - alpha) · M_dense(all N)

    이전 버전은 k 가 고정 top-1% 였음 (alpha 만 학습). 이제 k 도 슬라이드별
    entropy 따라 [0.1%, 1%] 사이에서 자동 결정 — ECSA dynamic-k + Feature_Adaptive
    gated pool 의 진정한 결합.
    """
    def __init__(self, k_min_pct=0.001, k_max_pct=0.01, gamma=1.0, entropy_method='v2', k_max_cap=0, inverse=False, entropy_only=False, entropy_k_floor=8,
                 loglinear=False, loglinear_k_min=8, loglinear_k_cap=500, loglinear_cap_frac=0.0,
                 inverse_threshold=1.0, hybrid_floor=False, hybrid_floor_alpha=0.0, hybrid_floor_min=8,
                 blend_w_bias=0.0, pure_cap=False, pure_cap_frac=0.0, pure_cap_min=8,
                 learnable_alpha=False, learnable_alpha_init=0.03, learnable_alpha_temp=20.0,
                 learnable_alpha_min=8, learnable_alpha_hybrid=False,
                 k_min_floor=4, k_max_floor=32):
        super().__init__()
        self.k_min_pct = k_min_pct
        self.k_max_pct = k_max_pct
        self.gamma = gamma
        self.entropy_method = entropy_method
        self.k_max_cap = k_max_cap
        self.inverse = inverse
        # inverse_threshold: 1.0 = always invert (legacy). <1.0 = adaptive, only invert when H_norm < threshold.
        # Recommended adaptive value: 0.3 (only flip on sharp-attention slides, where inverse helps).
        self.inverse_threshold = inverse_threshold
        # entropy_only=True: bypass k_min/k_max/k_max_cap; use k = round(exp(H(A))) (perplexity).
        # Hyperparameter-free, bag-size-independent. exp(H) = effective number of attended patches.
        self.entropy_only = entropy_only
        # entropy_k_floor: minimum k when entropy_only is on (prevents k=1 collapse on sharp-attention models)
        self.entropy_k_floor = entropy_k_floor
        # loglinear=True: k = k_min^(1-H_norm) * k_cap^H_norm  (N-independent geometric interpolation)
        # loglinear_cap_frac > 0: N-adaptive cap, k_cap_eff = min(k_cap, alpha*N). Helps small-bag datasets (e.g. CAM17 lymph nodes).
        self.loglinear = loglinear
        self.loglinear_k_min = loglinear_k_min
        self.loglinear_k_cap = loglinear_k_cap
        self.loglinear_cap_frac = loglinear_cap_frac
        # hybrid_floor=True: k = min(max(floor, exp(H)), alpha*N). Combines floor8's perplexity-driven k
        # with N-adaptive cap (best of both: cap-free perplexity + noise control on huge bags).
        self.hybrid_floor = hybrid_floor
        self.hybrid_floor_alpha = hybrid_floor_alpha
        self.hybrid_floor_min = hybrid_floor_min
        # pure_cap=True: k = alpha*N (entropy-free baseline). Used to test "is entropy actually useful,
        # or is k/N ratio alone enough?". Compares against floor8 (entropy-only) and hybrid_floor (both).
        self.pure_cap = pure_cap
        self.pure_cap_frac = pure_cap_frac
        self.pure_cap_min = pure_cap_min
        # learnable_alpha: differentiable cap fraction. alpha = sigmoid(alpha_logit).
        # k_target = alpha*N; soft top-k via sigmoid step (smooth), so gradient flows to alpha.
        # Eliminates per-cell sweep — model learns optimal cap from data.
        self.learnable_alpha = learnable_alpha
        self.learnable_alpha_temp = learnable_alpha_temp
        self.learnable_alpha_min = learnable_alpha_min
        # learnable_alpha_hybrid: extends LRA with hybrid_floor formula:
        #   k = min(max(floor, exp(H_use * log(N))), alpha*N)
        # where H_use = (1-H_norm) if H_norm < inverse_threshold (and inverse=True) else H_norm.
        # This keeps perplexity (entropy-driven natural k) AND learnable cap, AND adaptive inverse.
        self.learnable_alpha_hybrid = learnable_alpha_hybrid
        # ECC-DI: parameterized floors. Standard ECSA uses (4, 32). ECC-DI sets (8, 8) → strict k_min=8, no max-32 floor.
        self.k_min_floor = k_min_floor
        self.k_max_floor = k_max_floor
        if learnable_alpha:
            init = max(1e-4, min(0.999, learnable_alpha_init))
            import math as _m
            logit_init = _m.log(init / (1.0 - init))
            self.alpha_logit = nn.Parameter(torch.tensor(float(logit_init)))
        # Diagnostic stats accumulator (for hybrid mode): epoch-level training-set stats.
        # reset before each epoch via reset_diag_stats(); read via get_diag_stats().
        self._diag_stats = None
        # Adaptive-τ (warmup quantile): collect H_norms during warmup with inverse OFF,
        # then set self.inverse_threshold to the target quantile so sharp_ratio is consistent
        # across cells (cell-specific τ derived from data, not a global hyperparameter).
        self._in_warmup = False
        self._warmup_h_norms = []
        self._warmup_saved_threshold = None
        self._adaptive_tau_target = None  # set by core_utils.py if args.adaptive_tau is on
        # sparse_gate input: stats_vec(6) + attention stats(2: H_norm, k/N) = 8-dim
        # → attention sharpness까지 보고 sparse vs dense blend 결정 (옵션 A)
        self.sparse_gate = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0=fully dense, 1=fully sparse
        )
        # blend_w_bias: pre-sigmoid bias init. >0 shifts blend_w toward sparse at start of training.
        # Recommended: 1.0 → sigmoid(1)≈0.73 (start sparse-leaning).
        if blend_w_bias != 0.0:
            with torch.no_grad():
                self.sparse_gate[2].bias.fill_(blend_w_bias)

    def _entropy_k(self, A):
        """compute_dynamic_k_from_entropy 와 동일 로직 (k_min/max + entropy → k)"""
        N = A.shape[1]
        eps = 1e-8

        # pure_cap mode: k = alpha*N (entropy-free baseline for ablation).
        # Tests whether entropy/perplexity is needed at all, or k/N ratio alone suffices.
        if self.pure_cap and self.pure_cap_frac > 0:
            k = max(self.pure_cap_min, int(self.pure_cap_frac * N))
            k = min(k, max(1, (N - 1) // 2))
            return max(k, 1)

        # hybrid_floor mode: k = min(max(floor, exp(H)), alpha*N).
        # Combines floor8's cap-free perplexity with N-adaptive cap to limit noise on huge bags.
        if self.hybrid_floor:
            H_full = -torch.sum(A * torch.log(A + eps))
            if torch.isnan(H_full):
                return max(self.hybrid_floor_min, 1)
            k = int(round(torch.exp(H_full).item()))
            k = max(k, self.hybrid_floor_min)
            if self.hybrid_floor_alpha > 0:
                cap = max(self.hybrid_floor_min, int(self.hybrid_floor_alpha * N))
                k = min(k, cap)
            k = min(k, max(1, (N - 1) // 2))
            return max(k, 1)

        # entropy_only mode: pure perplexity. k = round(exp(H(A))). No k_min/k_max/cap.
        if self.entropy_only:
            H_full = -torch.sum(A * torch.log(A + eps))
            if torch.isnan(H_full):
                return max(self.entropy_k_floor, 1)
            k = int(round(torch.exp(H_full).item()))
            k = max(k, self.entropy_k_floor)        # ★ floor (default 8) — prevents k=1 collapse
            k = min(k, max(1, (N - 1) // 2))         # safety: ≤ half of N
            return max(k, 1)

        # loglinear mode: k = k_min^(1-H_norm) * k_cap^H_norm. Geometric interpolation.
        # If loglinear_cap_frac > 0: N-adaptive cap k_cap_eff = min(k_cap, cap_frac*N)
        # If self.inverse: low entropy (sharp) → large k (use 1-H_norm as exponent)
        # Adaptive inverse: if inverse_threshold < 1.0, only invert when H_norm < threshold
        if self.loglinear:
            H_full = -torch.sum(A * torch.log(A + eps))
            if torch.isnan(H_full) or N <= 1:
                return max(self.loglinear_k_min, 1)
            import math
            H_max = math.log(N)
            H_norm = max(0.0, min(1.0, (H_full / H_max).item()))
            if self.inverse and H_norm < self.inverse_threshold:
                H_use = 1.0 - H_norm
            else:
                H_use = H_norm
            k_cap_eff = self.loglinear_k_cap
            if self.loglinear_cap_frac > 0:
                k_cap_eff = min(self.loglinear_k_cap, max(self.loglinear_k_min, int(self.loglinear_cap_frac * N)))
            ratio = k_cap_eff / max(self.loglinear_k_min, 1)
            k = int(round(self.loglinear_k_min * (ratio ** H_use)))
            k = max(k, self.loglinear_k_min)
            k = min(k, k_cap_eff)
            k = min(k, max(1, (N - 1) // 2))
            return max(k, 1)

        k_min = max(self.k_min_floor, int(N * self.k_min_pct))
        k_max = max(self.k_max_floor, int(N * self.k_max_pct))
        # Bag-size-aware cap (paper finding: ECSA hurts on very large bags)
        if self.k_max_cap > 0 and k_max > self.k_max_cap:
            k_max = self.k_max_cap
        if k_max < k_min:
            k_max = k_min
        if self.entropy_method == 'v2':
            half_M = min(250, N // 2)
            if half_M < 1:
                return max(k_min, 1)
            top_vals = torch.topk(A, half_M, dim=1)[0]
            bot_vals = torch.topk(A, half_M, dim=1, largest=False)[0]
            sel = torch.cat([top_vals, bot_vals], dim=1)
            sel = sel / (sel.sum(dim=1, keepdim=True) + eps)
            M_total = sel.shape[1]
            H = -torch.sum(sel * torch.log(sel + eps))
            H_max = torch.log(torch.tensor(float(M_total), device=A.device))
        else:  # v1: full Shannon
            H = -torch.sum(A * torch.log(A + eps))
            H_max = torch.log(torch.tensor(float(N), device=A.device))
        if torch.isnan(H):
            return k_min
        H_norm = (H / H_max).clamp(0.0, 1.0)
        # Forward: high entropy → large k.  Inverse: low entropy → large k.
        # ECC-DI: threshold-based dynamic inverse (only flip when H_norm < inverse_threshold).
        # Legacy behavior (always flip when self.inverse=True) preserved when inverse_threshold==1.0 (default).
        H_norm_val = H_norm.item()
        if self.inverse and H_norm_val < self.inverse_threshold:
            H_use = 1.0 - H_norm_val
        else:
            H_use = H_norm_val
        k = int(k_min + (k_max - k_min) * (H_use ** self.gamma))
        k = min(k, max(1, (N - 1) // 2))
        return max(k, 1)

    def forward(self, A, h, stats_vec):
        # Dense aggregation (전체 N)
        M_dense = torch.mm(A, h)
        N = A.shape[1]

        if self.learnable_alpha:
            # ── Learnable α (1D scalar via sigmoid) ────────────────────────
            alpha = torch.sigmoid(self.alpha_logit)

            if self.learnable_alpha_hybrid:
                # ── HYBRID mode: k = min(max(floor, exp(H_use·log N)), α·N) ──
                # exact same shape as hybrid_floor sweep, but α is learnable + adaptive inverse on H_use
                eps = 1e-8
                H_full = -torch.sum(A * torch.log(A + eps))
                log_N_t = torch.log(torch.tensor(float(N), device=A.device, dtype=A.dtype))
                if torch.isnan(H_full) or N <= 1:
                    H_norm_val = 0.5
                else:
                    H_norm_val = (H_full / log_N_t).clamp(0.0, 1.0).item()
                # adaptive_tau warmup: collect H_norms (inverse forcibly disabled during warmup)
                if self._in_warmup:
                    self._warmup_h_norms.append(H_norm_val)
                # adaptive inverse (only flip when sharp)
                is_sharp = self.inverse and self.inverse_threshold < 1.0 and H_norm_val < self.inverse_threshold
                H_use_val = (1.0 - H_norm_val) if is_sharp else H_norm_val
                # k_entropy = exp(H_use * log N) — log-domain for numerical stability (NOT N**H_use)
                H_use_t = torch.tensor(H_use_val, device=A.device, dtype=A.dtype)
                k_entropy = torch.exp(H_use_t * log_N_t)
                floor_t = torch.tensor(float(self.learnable_alpha_min), device=A.device, dtype=A.dtype)
                # natural k: max(floor, k_entropy) — constant per slide (no α gradient here)
                natural_k = torch.maximum(floor_t, k_entropy)
                # cap: α·N — α gradient flows ONLY when cap is binding (cap < natural_k)
                cap_alpha_N = alpha * float(N)
                k_target = torch.minimum(natural_k, cap_alpha_N)
                # Track diagnostics (sharp_ratio, cap_binding_ratio, H_norm, k, k/N)
                if self._diag_stats is not None:
                    cap_binding = (cap_alpha_N.detach() < natural_k.detach()).item()
                    self._diag_stats['count'] += 1
                    self._diag_stats['sharp'] += int(is_sharp)
                    self._diag_stats['cap_binding'] += int(cap_binding)
                    self._diag_stats['H_norm_sum'] += H_norm_val
                    self._diag_stats['k_sum'] += float(k_target.detach().item())
                    self._diag_stats['kN_sum'] += float(k_target.detach().item()) / max(N, 1)
            else:
                # ── ORIGINAL Stage 2 LRA: k = α·N + softplus floor (entropy-free) ──
                k_target_raw = alpha * float(N)
                min_k = float(self.learnable_alpha_min)
                k_target = k_target_raw + F.softplus(min_k - k_target_raw)
            sorted_vals, sorted_idx = torch.sort(A, dim=1, descending=True)
            ranks = torch.arange(N, device=A.device, dtype=A.dtype).unsqueeze(0)  # [1,N]
            soft_mask_sorted = torch.sigmoid((k_target - ranks - 0.5) * self.learnable_alpha_temp)
            soft_mask = torch.zeros_like(A)
            soft_mask.scatter_(1, sorted_idx, soft_mask_sorted)
            A_soft = A * soft_mask
            A_sparse = A_soft / (A_soft.sum(dim=1, keepdim=True) + 1e-8)
            M_sparse = torch.mm(A_sparse, h)
            # Effective k for monitoring (sum of soft mask). Use detached int for downstream.
            k = int(round(soft_mask.sum().item()))
            k = max(1, min(k, N))
        else:
            # Entropy-driven k_i + hard top-k aggregation
            k = min(self._entropy_k(A), N)
            top_k_vals, top_k_idx = torch.topk(A, k, dim=1)
            A_sparse = torch.zeros_like(A)
            A_sparse.scatter_(1, top_k_idx, top_k_vals)
            A_sparse = A_sparse / (A_sparse.sum(dim=1, keepdim=True) + 1e-8)
            M_sparse = torch.mm(A_sparse, h)
        # Gate-weighted blend (옵션 A: attention 통계 포함)
        # 추가 features: H_norm(A) [0,1], k/N [0,1] — slide-specific attention sharpness
        eps = 1e-8
        H_full = -torch.sum(A * torch.log(A + eps))
        if torch.isnan(H_full) or N <= 1:
            H_norm = 0.0
        else:
            import math
            H_norm = max(0.0, min(1.0, (H_full / math.log(N)).item()))
        attn_stats = torch.tensor([H_norm, float(k) / max(N, 1)],
                                   device=stats_vec.device, dtype=stats_vec.dtype)
        extended_stats = torch.cat([stats_vec, attn_stats])  # dim 6 + 2 = 8
        blend_w = self.sparse_gate(extended_stats)  # scalar in [0,1]
        M = blend_w * M_sparse + (1 - blend_w) * M_dense
        return M, k, blend_w.item()

    def get_learned_alpha(self):
        """Returns current sigmoid(alpha_logit) for logging. Only valid when learnable_alpha=True."""
        if not self.learnable_alpha:
            return None
        with torch.no_grad():
            return torch.sigmoid(self.alpha_logit).item()

    def reset_diag_stats(self):
        """Reset per-epoch diagnostic stats (call at start of each epoch)."""
        self._diag_stats = {
            'count': 0, 'sharp': 0, 'cap_binding': 0,
            'H_norm_sum': 0.0, 'k_sum': 0.0, 'kN_sum': 0.0,
        }

    def get_diag_stats(self):
        """Returns dict of per-epoch ratios. None if no stats accumulated."""
        if self._diag_stats is None or self._diag_stats['count'] == 0:
            return None
        s = self._diag_stats
        n = s['count']
        return {
            'count': n,
            'sharp_ratio': s['sharp'] / n,
            'cap_binding_ratio': s['cap_binding'] / n,
            'mean_H_norm': s['H_norm_sum'] / n,
            'mean_k': s['k_sum'] / n,
            'mean_kN_ratio': s['kN_sum'] / n,
        }

    def start_warmup(self, force_no_inverse=True):
        """Begin adaptive-τ warmup: collect H_norms.

        force_no_inverse=True (default, original semantics): temporarily set
            inverse_threshold=0.0 so the inverse flip never triggers during
            collection. Use this for the FIRST τ computation, where any
            existing inverse_threshold value is meaningless / arbitrary.

        force_no_inverse=False: leave inverse_threshold at its current (already-
            tuned) value while collecting. Use this for PERIODIC re-updates
            after the first τ has been set, so the model continues to behave
            in its trained regime while we sample the latest H_norm distribution.
        """
        self._in_warmup = True
        self._warmup_h_norms = []
        if force_no_inverse:
            # Save and disable inverse_threshold (set to 0.0 so H_norm < 0.0 is never True).
            self._warmup_saved_threshold = self.inverse_threshold
            self.inverse_threshold = 0.0
        else:
            self._warmup_saved_threshold = None  # nothing to restore

    def finalize_warmup(self, target_quantile=0.3):
        """Compute τ as the target quantile of collected H_norms; restore inverse path.

        Returns τ (float) actually set, or None if no H_norms were collected
        (in which case the saved threshold is restored unchanged).
        """
        self._in_warmup = False
        if not self._warmup_h_norms:
            self.inverse_threshold = self._warmup_saved_threshold
            self._warmup_saved_threshold = None
            return None
        try:
            import numpy as _np
            tau = float(_np.quantile(_np.asarray(self._warmup_h_norms, dtype=float), target_quantile))
        except Exception:
            arr = sorted(self._warmup_h_norms)
            idx = int(round(target_quantile * (len(arr) - 1)))
            tau = float(arr[max(0, min(len(arr) - 1, idx))])
        # Clamp to (0,1) so inverse remains a real "low-entropy slide" gate.
        tau = max(1e-3, min(0.999, tau))
        self.inverse_threshold = tau
        # Free the buffer; keep saved_threshold for debugging.
        self._warmup_h_norms = []
        return tau

    def is_in_warmup(self):
        return self._in_warmup

    def get_sparse_ratio(self, stats_vec, A=None, k=None, N=None):
        """sparse vs dense blend ratio. Forward 외 호출 시 attention 정보 없으면 0으로 채움."""
        with torch.no_grad():
            if A is not None and k is not None and N is not None and N > 1:
                eps = 1e-8
                H_full = -torch.sum(A * torch.log(A + eps))
                import math
                H_norm = max(0.0, min(1.0, (H_full / math.log(N)).item())) if not torch.isnan(H_full) else 0.0
                attn_stats = torch.tensor([H_norm, float(k) / N],
                                           device=stats_vec.device, dtype=stats_vec.dtype)
            else:
                # Fallback: 호출 시 attention 없으면 0으로 (학습 외부 monitoring용)
                attn_stats = torch.zeros(2, device=stats_vec.device, dtype=stats_vec.dtype)
            extended = torch.cat([stats_vec, attn_stats])
            return self.sparse_gate(extended).item()


class UncertaintyLossWeight(nn.Module):
    """
    Kendall et al. (CVPR 2018) - Multi-Task Learning Using Uncertainty to Weigh Losses

    Homoscedastic uncertainty 기반 adaptive loss weighting.
    bag_loss와 instance_loss의 비중을 학습 과정에서 자동으로 조절한다.

    Loss = (1 / 2*sigma_bag^2) * L_bag + log(sigma_bag)
         + (1 / 2*sigma_inst^2) * L_inst + log(sigma_inst)

    log_var = log(sigma^2) 를 learnable parameter로 사용하여 수치 안정성 확보.
    """
    def __init__(self):
        super().__init__()
        # log(sigma^2) 초기화: 0 → sigma=1 → weight=0.5 (균등 시작)
        self.log_var_bag = nn.Parameter(torch.zeros(1))
        self.log_var_inst = nn.Parameter(torch.zeros(1))

    def forward(self, bag_loss, inst_loss):
        # precision (inverse variance) as weight
        w_bag = torch.exp(-self.log_var_bag)
        w_inst = torch.exp(-self.log_var_inst)

        # uncertainty-weighted total loss with regularization
        total_loss = 0.5 * w_bag * bag_loss + 0.5 * self.log_var_bag + \
                     0.5 * w_inst * inst_loss + 0.5 * self.log_var_inst

        return total_loss

    def get_weights(self):
        """현재 학습된 weight 반환 (로깅용)"""
        w_bag = torch.exp(-self.log_var_bag).item()
        w_inst = torch.exp(-self.log_var_inst).item()
        # normalize to sum to 1 for interpretability
        w_total = w_bag + w_inst
        return {
            'w_bag': w_bag / w_total,
            'w_inst': w_inst / w_total,
            'w_bag_raw': w_bag,
            'w_inst_raw': w_inst,
            'log_var_bag': self.log_var_bag.item(),
            'log_var_inst': self.log_var_inst.item(),
        }


def compute_dynamic_k(A, k_min=4, k_max=16, gamma=1.0, method='v1', confidence=None,
                      inverse=False, adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
    """
    Attention 엔트로피 기반 dynamic k 계산

    Args:
        A: softmax attention [1, N] or [N]
        k_min: 최소 k (adaptive_k_range=False일 때 사용)
        k_max: 최대 k (adaptive_k_range=False일 때 사용)
        gamma: 비선형 매핑 파라미터 (>1이면 sharp쪽에 더 민감)
        method: 'v1' (H/log(N)), 'v2' (top-M 기반, patch 수 bias 제거)
        confidence: 미사용 (호환성 유지)
        inverse: True이면 역방향 매핑 (sharp→k 큼, flat→k 작음)
        adaptive_k_range: True이면 k_min/k_max를 패치 수 N에 비례하여 자동 설정
        k_min_pct: adaptive 시 k_min = max(4, N * k_min_pct) (기본 0.1%)
        k_max_pct: adaptive 시 k_max = max(32, N * k_max_pct) (기본 1%)

    Returns:
        k: dynamic k value
        entropy_norm: 정규화된 엔트로피 [0, 1]
        entropy_raw: raw 엔트로피 값
    """
    if len(A.shape) == 1:
        A = A.unsqueeze(0)

    N = A.shape[1]
    eps = 1e-8

    # Adaptive k range: 패치 수에 비례하여 k_min/k_max 자동 설정
    if adaptive_k_range:
        k_min = max(4, int(N * k_min_pct))
        k_max = max(32, int(N * k_max_pct))

    if method == 'v2':
        # ── Top-250 + Bottom-250 기반 엔트로피: 양 극단으로 분포 형태 포착 ──
        half_M = min(250, N // 2)
        top_vals = torch.topk(A, half_M, dim=1)[0]           # [1, half_M]
        bottom_vals = torch.topk(A, half_M, dim=1, largest=False)[0]  # [1, half_M]
        selected = torch.cat([top_vals, bottom_vals], dim=1)  # [1, 2*half_M]
        selected = selected / selected.sum(dim=1, keepdim=True)  # 재정규화
        M_total = selected.shape[1]
        H = -torch.sum(selected * torch.log(selected + eps))
        H_max = torch.log(torch.tensor(float(M_total), device=A.device))
    else:  # v1 (기존) + v3 fallback
        H = -torch.sum(A * torch.log(A + eps))
        H_max = torch.log(torch.tensor(float(N), device=A.device))

    # NaN 방어
    if torch.isnan(H):
        return k_min, 0.0, 0.0

    H_norm = (H / H_max).clamp(0.0, 1.0)

    if inverse:
        k = int(k_min + (k_max - k_min) * ((1.0 - H_norm.item()) ** gamma))
    else:
        k = int(k_min + (k_max - k_min) * (H_norm.item() ** gamma))

    # 안전 조건: 2k <= N (positive k개 + negative k개)
    k = min(k, (N - 1) // 2)
    k = max(k, 1)

    return k, H_norm.item(), H.item()

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        learnable_temp=False, adaptive_temp=False, sparse_topk=False,
        attn_norm='softmax', use_gelu=False, adaptive_sparse_pool=False,
        feature_adaptive=False, k_max_cap=0, dk_inverse=False, entropy_only=False, entropy_k_floor=8,
        loglinear=False, loglinear_k_min=8, loglinear_k_cap=500, loglinear_cap_frac=0.0,
        k_min_pct=0.001, k_max_pct=0.01, gamma=1.0, entropy_method='v2',
        inverse_threshold=1.0, hybrid_floor=False, hybrid_floor_alpha=0.0, hybrid_floor_min=8,
        blend_w_bias=0.0, pure_cap=False, pure_cap_frac=0.0, pure_cap_min=8,
        learnable_alpha=False, learnable_alpha_init=0.03, learnable_alpha_temp=20.0, learnable_alpha_min=8,
        learnable_alpha_hybrid=False,
        lsap_temp=False, lsap_eps=0.01):
        super().__init__()
        # Attention normalization choice (Phase 2 baseline 비교용):
        #   'softmax'   : 표준 softmax (기본)
        #   'sparsemax' : Martins & Astudillo 2016 (entmax 패키지)
        #   'entmax15'  : Peters et al. 2019, alpha=1.5
        self.attn_norm = attn_norm
        # feature_adaptive 가 True 이면 5개 adaptive 모듈 모두 활성화
        # (norm + activation + sparse_pool + temp; analyzer 는 항상 같이 사용)
        self.feature_adaptive = feature_adaptive
        self.adaptive_sparse_pool = adaptive_sparse_pool or feature_adaptive
        if feature_adaptive:
            adaptive_temp = True  # force adaptive temperature
        if self.adaptive_sparse_pool or feature_adaptive:
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
        if feature_adaptive:
            self.adaptive_norm = AdaptiveNormalization(embed_dim)
            self.adaptive_act = AdaptiveActivation()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        # GELU vs ReLU: ReLU 는 negative values 를 죽임 → CTransPath 같은 zero-centered
        # features (mean≈0, min≈-1.1) 의 정보 절반 손실. GELU 는 이를 보존.
        # 사용자 옛 슬라이드 결과 (CTransPath +2.62%) 의 핵심 원인 추정.
        # feature_adaptive 모드에서는 AdaptiveActivation 이 처리하므로 Identity 사용
        if feature_adaptive:
            activation = nn.Identity()
        else:
            activation = nn.GELU() if use_gelu else nn.ReLU()
        fc = [nn.Linear(size[0], size[1]), activation, nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # Top-K Sparse Attention Pooling: k가 최종 aggregation에 직접 영향
        self.sparse_topk = sparse_topk

        # Temperature 모드: adaptive > learnable > fixed (1.0)
        # lsap_temp 옵션: AdaptiveTemperature 대신 LSAPTemperature (4-dim stats: mean,std,max,log N)
        self.adaptive_temp = adaptive_temp
        self.learnable_temp = learnable_temp
        self.lsap_temp = lsap_temp
        if adaptive_temp:
            if lsap_temp:
                self.temp_module = LSAPTemperature(eps=lsap_eps)
            else:
                self.temp_module = AdaptiveTemperature(n_attn_heads=1)
        elif learnable_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier, k_sample=None):
        k = k_sample if k_sample is not None else self.k_sample
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier, k_sample=None):
        k = k_sample if k_sample is not None else self.k_sample
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def get_temperature(self):
        """현재 global temperature 값 반환 (로깅용)"""
        if self.learnable_temp and not self.adaptive_temp:
            return torch.exp(self.log_temp).item()
        return 1.0

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False,
                dynamic_k=False, k_min=4, k_max=16, gamma=1.0, dk_method='v1', dk_inverse=False,
                adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
        # ── Phase 2: feature-adaptive normalization (입력 단계) ──
        stats_vec = None
        if self.feature_adaptive:
            stats_vec = self.feat_analyzer(h)
            h = self.adaptive_norm(h, stats_vec)
        elif self.adaptive_sparse_pool:
            stats_vec = self.feat_analyzer(h)

        A, h = self.attention_net(h)  # NxK

        # ── Phase 2: feature-adaptive activation (attention_net 출력 후) ──
        if self.feature_adaptive:
            h = self.adaptive_act(h, stats_vec)

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A

        # Temperature scaling: adaptive (per-slide) > learnable (global) > fixed (1.0)
        if self.adaptive_temp:
            temp = self.temp_module(A_raw)
            A_scaled = A / temp
        elif self.learnable_temp:
            temp = torch.exp(self.log_temp)
            A_scaled = A / temp
        else:
            temp = None
            A_scaled = A

        # Attention normalization: softmax (default) / sparsemax / entmax15
        attn_norm = getattr(self, 'attn_norm', 'softmax')
        if attn_norm == 'sparsemax':
            from entmax import sparsemax
            A = sparsemax(A_scaled, dim=1)
        elif attn_norm == 'entmax15':
            from entmax import entmax15
            A = entmax15(A_scaled, dim=1)
        else:  # 'softmax'
            A = F.softmax(A_scaled, dim=1)

        # Dynamic k 계산 — sparse_topk 모드에서는 instance_eval 없어도 항상 계산
        if dynamic_k and (instance_eval or self.sparse_topk):
            k_sample, entropy_norm, entropy_raw = compute_dynamic_k(
                A, k_min, k_max, gamma, method=dk_method, confidence=None, inverse=dk_inverse,
                adaptive_k_range=adaptive_k_range, k_min_pct=k_min_pct, k_max_pct=k_max_pct)
        else:
            k_sample = self.k_sample
            entropy_norm = None
            entropy_raw = None

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier, k_sample=k_sample)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier, k_sample=k_sample)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        # ── Aggregation: adaptive_sparse_pool > sparse top-k > full attention ──
        bag_size_N = A.shape[1]
        sparse_ratio = None
        if self.adaptive_sparse_pool:
            # Reuse stats_vec computed at the input stage if available
            if stats_vec is None:
                stats_vec = self.feat_analyzer(h)
            M, k_pool, sparse_ratio = self.adaptive_pool(A, h, stats_vec)
        elif self.sparse_topk and dynamic_k:
            # Top-k 패치만으로 aggregation → k가 최종 logit에 직접 영향
            k_pool = min(k_sample, A.shape[1])
            top_k_vals, top_k_idx = torch.topk(A, k_pool, dim=1)  # [1, k_pool]
            A_sparse = torch.zeros_like(A)
            A_sparse.scatter_(1, top_k_idx, top_k_vals)
            A_sparse = A_sparse / (A_sparse.sum(dim=1, keepdim=True) + 1e-8)  # 재정규화
            M = torch.mm(A_sparse, h)
        else:
            k_pool = bag_size_N  # baseline: 모든 patch 사용
            M = torch.mm(A, h)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        # Always log bag size N + selected k_pool (top-k aggregation count) for visibility
        results_dict.update({
            'num_patches': bag_size_N,
            'k_pool': k_pool,
        })
        if dynamic_k:
            results_dict.update({
                'dynamic_k': k_sample,
                'entropy_norm': entropy_norm,
                'entropy_raw': entropy_raw,
            })
        if self.adaptive_temp or self.learnable_temp:
            results_dict.update({'temperature': temp.item() if torch.is_tensor(temp) else temp})
        if self.adaptive_sparse_pool and sparse_ratio is not None:
            results_dict.update({'sparse_ratio': sparse_ratio})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        learnable_temp=False, adaptive_temp=False, sparse_topk=False,
        attn_norm='softmax', use_gelu=False):
        nn.Module.__init__(self)
        self.attn_norm = attn_norm
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        activation = nn.GELU() if use_gelu else nn.ReLU()
        fc = [nn.Linear(size[0], size[1]), activation, nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # Top-K Sparse Attention Pooling
        self.sparse_topk = sparse_topk

        # Temperature 모드
        self.adaptive_temp = adaptive_temp
        self.learnable_temp = learnable_temp
        if adaptive_temp:
            self.temp_module = AdaptiveTemperature(n_attn_heads=n_classes)
        elif learnable_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False,
                dynamic_k=False, k_min=4, k_max=16, gamma=1.0, dk_method='v1', dk_inverse=False,
                adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A

        # Temperature scaling: adaptive (per-slide) > learnable (global) > fixed (1.0)
        if self.adaptive_temp:
            temp = self.temp_module(A_raw)
            A_scaled = A / temp
        elif self.learnable_temp:
            temp = torch.exp(self.log_temp)
            A_scaled = A / temp
        else:
            temp = None
            A_scaled = A

        # Attention normalization: softmax (default) / sparsemax / entmax15
        attn_norm = getattr(self, 'attn_norm', 'softmax')
        if attn_norm == 'sparsemax':
            from entmax import sparsemax
            A = sparsemax(A_scaled, dim=1)
        elif attn_norm == 'entmax15':
            from entmax import entmax15
            A = entmax15(A_scaled, dim=1)
        else:  # 'softmax'
            A = F.softmax(A_scaled, dim=1)

        # Dynamic k 계산 (CLAM_MB: label class의 attention branch 사용)
        if dynamic_k and (instance_eval or self.sparse_topk):
            if label is not None:
                cls_idx = label.item()
            else:
                cls_idx = A.sum(dim=1).argmax().item()
            A_cls = A[cls_idx].unsqueeze(0)  # [1, N]
            k_sample, entropy_norm, entropy_raw = compute_dynamic_k(
                A_cls, k_min, k_max, gamma, method=dk_method, confidence=None, inverse=dk_inverse,
                adaptive_k_range=adaptive_k_range, k_min_pct=k_min_pct, k_max_pct=k_max_pct)
        else:
            k_sample = self.k_sample
            entropy_norm = None
            entropy_raw = None

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier, k_sample=k_sample)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier, k_sample=k_sample)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        # ── Aggregation: sparse top-k vs full attention (per head) ──
        if self.sparse_topk and dynamic_k:
            k_pool = min(k_sample, A.shape[1])
            A_sparse = torch.zeros_like(A)
            for head_idx in range(A.shape[0]):
                top_k_vals, top_k_idx = torch.topk(A[head_idx].unsqueeze(0), k_pool, dim=1)
                A_sparse[head_idx].scatter_(0, top_k_idx.squeeze(0), top_k_vals.squeeze(0))
            A_sparse = A_sparse / (A_sparse.sum(dim=1, keepdim=True) + 1e-8)
            M = torch.mm(A_sparse, h)
        else:
            M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        if dynamic_k:
            results_dict.update({
                'dynamic_k': k_sample,
                'entropy_norm': entropy_norm,
                'entropy_raw': entropy_raw,
                'num_patches': A.shape[1],
            })
        if self.adaptive_temp or self.learnable_temp:
            results_dict.update({'temperature': temp.item() if torch.is_tensor(temp) else temp})
        return logits, Y_prob, Y_hat, A_raw, results_dict
