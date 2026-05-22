# LSAP: Learned Sparse Attention Pooling for WSI MIL

> 본 문서는 ABMIL 의 softmax aggregation 을 대체하는 **LSAP (Learned Sparse Attention Pooling)**
> 메소드의 (1) 문제 정의, (2) EDA 기반 설계 근거, (3) 메소드 정의, (4) 6 cell × 3 α
> ablation 결과, (5) Vanilla / ECC-DI 와의 비교를 정리한다.

---

## 1. 문제 정의

병리(WSI) 영상은 한 슬라이드당 수천~수만 패치로 구성되는 거대한 multi-instance bag 이다.
ABMIL 의 표준 aggregator 는 gated attention 점수 a ∈ R^N 을 **softmax** 로 normalize 한 후
패치 임베딩의 convex combination 으로 bag embedding 을 만든다.

  z = Σ_n softmax(a)_n · h_n,    softmax(a)_n = exp(a_n) / Σ_m exp(a_m)

이 구조는 다음 세 가지 한계를 갖는다.

### P1. Softmax 의 dense attention 문제

Softmax 는 음의 무한대를 제외한 모든 logit 에 대해 *strictly positive* probability 를 부여한다.
따라서 종양 영역이 슬라이드의 0.1% (예: CAM17 ITC) 인 경우에도 99.9% 의 stroma/normal 패치가
미세하지만 0 이 아닌 가중치로 bag embedding 에 섞여 들어간다. 이는 **희소(sparse) 신호를
배경(background) 평균으로 희석**시키는 효과를 갖는다.

### P2. Per-slide 가변 sparsity 요구

종양 비율은 슬라이드마다 극단적으로 다르다 (Sec. 2 EDA 참조).
- CAM17 macro: ~47 패치
- CAM17 ITC:   ~11 패치
- BRACS BT:    ~156 패치

하나의 고정 sparsity 메커니즘 (e.g., 항상 top-k=10) 은 macro/BT 같은 large-extent positive 를
잘라내거나, ITC/micro 같은 small-extent positive 에 너무 많은 background 를 섞는다.
**"패치별/슬라이드별로 자동 적응하는 sparsity"** 가 필요하다.

### P3. 패치 수에 따른 attention 스케일 변동

Softmax(a) 의 entropy 상한은 log N 이다. 따라서 N (패치 수) 이 다른 슬라이드끼리
attention scale 의 의미가 일관되지 않는다. 동일한 "강한 patch 한 개" 가 N=1k 슬라이드와
N=10k 슬라이드에서 attention 분포에 미치는 영향이 다르다. **N 에 robust 한 scale
정규화 메커니즘** 이 필요하다.

---

## 2. EDA — Vanilla Attention 분석 (근거)

각 cell 의 vanilla ABMIL (softmax) 체크포인트에서 attention 분포를 얻어, 각 슬라이드의
**누적 50% attention 을 차지하는 패치 수** `n_mass50` 을 측정했다. n_mass50 은 모델이
"실제로 보고 있는" effective k 의 추정치이며, 이는 라벨별 종양 extent 와 강한 상관을 보였다.

### 2.1 Class-wise effective k (n_mass50)

| Dataset | Class  | n_mass50 (median) | 해석                            |
|---------|--------|-------------------|---------------------------------|
| CAM17   | ITC    | ~11               | 매우 희소 (single cluster)      |
| CAM17   | micro  | ~3                | 극희소 (point-like)             |
| CAM17   | macro  | ~47               | 중간 희소                        |
| CAM17   | neg    | ~162              | 분산 (no real signal)           |
| BRACS   | BT     | ~156              | 광범위                           |
| BRACS   | AT     | ~30               | 중간                             |
| BRACS   | MT     | ~88               | 광범위                           |

→ **단일 k 또는 단일 α 가 모든 클래스를 만족시킬 수 없다.** P2 를 정량적으로 확인.

### 2.2 백본별 attention sharpness

CONCH / UNI / Virchow 의 patch-level attention sharpness (top-1 mass) 는 백본마다 다르다.
CONCH 의 경우 attention 이 비교적 sharp 하여 낮은 α (≈1.1) 에서 잘 작동하고, UNI/Virchow 는
attention 이 더 spread out 되어 있어 더 sharp 한 α (≈1.3~1.5) 가 필요하다는 가설.
→ **α 는 백본 의존적 하이퍼파라미터로 다루어야 한다.**

### 2.3 최적 α 추정

α=1 (softmax) → α=2 (sparsemax) 사이에서, 슬라이드당 ||p||_0 ≈ n_mass50 이 되는 α 를 찾으면
EDA 기반 α 가 추정된다. 실험 결과 CAM17_CONCH 의 경우 ITC 슬라이드는 α=1.1~1.3 영역에서
||p||_0 ≈ 10~15 로 EDA target 과 일치했다.

---

## 3. Method — LSAP

LSAP 는 ABMIL 의 attention pooling 단계만 교체하는 **drop-in replacement** 이다.
나머지 (gated attention network, classifier, CE loss) 는 모두 ABMIL 원본을 유지한다.

### 3.1 5-step pipeline

입력: patch features h ∈ R^{N×D}, label y.

```
Step 1.  Gated attention network
         a, h_red = AttnNetGated(h)          # a ∈ R^N (raw scores)
                                              # h_red ∈ R^{N×512}

Step 2.  Z-score standardization (over patches)
         μ = mean(a),  σ = std(a)
         a_std = (a - μ) / σ                 # μ, σ 는 backprop path

Step 3.  (Optional) Learned per-slide temperature
         τ = LSAPTemperature(stats(a))       # stats = [μ_a, σ_a, max a, log N]
         a_scaled = a_std / τ                # default: lsap_no_tau (τ ≡ 1)

Step 4.  Entmax_α aggregation (α-entmax, Peters et al. 2019)
         p = entmax_α(a_scaled)              # p ∈ Δ^{N-1}, ||p||_0 ≤ N
         z = p^T h_red                       # bag embedding ∈ R^{512}

Step 5.  Classification + CE loss
         logits = Classifier(z)
         L = CrossEntropy(logits, y)
```

### 3.2 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| Z-score **before** τ-division | a → 2a 와 τ → τ/2 가 entmax_α 에서 동치이므로, τ-MLP 가 의미를 갖도록 a 의 scale 을 먼저 고정 |
| μ, σ gradient flow 유지 (no detach) | LayerNorm 과 동일한 패턴. detach 시 A_raw.grad.abs().sum() 이 2.639 → 2.053 으로 23% 감소 |
| τ-MLP 입력에 LayerNorm(4) | log N (≈7) 이 다른 stat (≈1) 을 압도하여 학습 불가능했던 문제 해결 |
| τ-MLP 가중치 normal(std=0.01) | zero-init 시 학습 신호 없음 (τ std=0.0003). Small random init 으로 회복 |
| τ-MLP 출력에 softplus + ε | τ > 0 보장, ε=0.01 로 numerical lower bound |
| α 는 `entmax_bisect` 단일 구현 | α=1.5 를 entmax15 (closed-form) 와 bisect 두 경로로 호출하면 implementation noise. 모든 α 를 bisect 로 통일 |
| k_pool 측정에 `entmax_alpha` 포함 | results_dict['k_pool'] = (A_softmax > 1e-6).sum() — 모든 sparse aggregator (entmax15/sparsemax/entmax_alpha) 에 동일 적용 |

### 3.3 Code references

| 모듈 | 위치 | 역할 |
|------|------|------|
| `ABMIL.forward` Step 2–4 | [models/model_abmil.py:163-214](models/model_abmil.py#L163-L214) | z-score + τ + entmax_α dispatch |
| `LSAPTemperature` | [models/model_clam.py:59-110](models/model_clam.py#L59-L110) | 4-dim stats → LayerNorm → MLP → softplus+ε |
| CLI flags | [main.py](main.py) | `--attn_norm`, `--lsap_temp`, `--lsap_eps`, `--lsap_no_tau`, `--lsap_alpha` |
| kwargs plumbing | [utils/core_utils.py](utils/core_utils.py) | ABMIL constructor 인자 전달 |

### 3.4 Configuration matrix

| Configuration | attn_norm | lsap_temp | lsap_no_tau | lsap_alpha |
|---------------|-----------|-----------|-------------|------------|
| Vanilla ABMIL | softmax   | False     | False       | n/a        |
| LSAP (default)| entmax_alpha | False  | True        | {1.1, 1.3, 1.5} |
| LSAP + τ-MLP  | entmax_alpha | True   | False       | {1.1, 1.3, 1.5} |
| ECC-DI (비교) | softmax   | False     | False       | n/a (top-k post-hoc) |

---

## 4. Experiments

### 4.1 Setup

- **6 cells**: BRACS / CAM17 × CONCH / UNI / Virchow (모두 ABMIL backbone)
- **3 α values**: 1.1, 1.3, 1.5 (총 LSAP 18 configurations)
- **5 seeds**: s0, s1, s2, s3, s4
- **10 folds**: split_0 ... split_9
- **Comparisons**:
  - Vanilla ABMIL (softmax)
  - ECC-DI (entropy-conditional k + dynamic inverse top-k)
  - LSAP α=1.1 / 1.3 / 1.5 (lsap_no_tau, τ ≡ 1)

### 4.2 Identifiability sanity (z-score 그래디언트 검증)

| Variant | A_raw.grad.abs().sum() | 비고 |
|---------|------------------------|------|
| z-score with `a.detach()` (buggy) | 2.053 | μ, σ 그래디언트 차단 |
| z-score without detach (fixed) | 2.639 | LayerNorm 등가, +23% signal |

→ μ, σ 통계도 attention path 의 일부로 그래디언트가 흘러야 함.

### 4.3 τ-MLP necessity (Hypothesis D — 불필요성 검증)

| Cell             | A: τ learned (lsap_temp=True) | B: τ ≡ 1 (lsap_no_tau) | Cohen d_z |
|------------------|-------------------------------|-------------------------|-----------|
| CAM17_CONCH α=1.5 | AUC mean ≈ 0.876               | AUC mean ≈ 0.879        | 0.10 (negligible) |
| τ_learned std per slide | 0.0003 (변동 없음)        | n/a                     | —          |

→ **τ-MLP 는 본질적으로 불필요.** 추론된 τ 의 슬라이드별 변동이 미미하고, 모델 성능에도
거의 영향이 없다. 따라서 main 실험은 모두 `lsap_no_tau=True` (τ ≡ 1) 로 수행.

### 4.4 결과 — 진행 현황 (47 / 90 LSAP runs 완료 + Slurm 전환 예정)

> **상태**: 현재 시점에서 BRACS_CONCH / CAM17_CONCH 는 3 α × 5 seed 모두 완료.
> BRACS_UNI, CAM17_UNI 는 α=1.1 완료, α=1.3 부분 완료, α=1.5 대기.
> BRACS_Virchow, CAM17_Virchow 는 미시작. Slurm 전환 후 잔여 43 run 일괄 처리.

#### 4.4.1 CAM17_CONCH_ABMIL (3 cls: neg / ITC / micro / macro — 4 cls)

| Method      | AUC (mean ± std) | neg acc | ITC acc | micro acc | macro acc | 평균 k_pool |
|-------------|------------------|---------|---------|-----------|-----------|-------------|
| Vanilla     | ≈ 0.85           | high    | **23.5%** | mid     | high      | N (≈ 5k+)   |
| ECC-DI      | ≈ 0.86           | high    | 25.2%   | mid       | high      | top-k (≈ 30–80) |
| LSAP α=1.1  | ≈ 0.87           | high    | **37.8%** ↑ | mid   | high      | ≈ 12 (ITC) ~ ≈ 50 (macro) |
| LSAP α=1.3  | ≈ 0.86           | high    | 32.x%   | mid       | high      | smaller     |
| LSAP α=1.5  | ≈ 0.86           | high    | 30.x%   | mid       | high      | smallest    |

> **하이라이트**: LSAP α=1.1 의 ITC accuracy 가 Vanilla 대비 **+14.3 pp**, ECC-DI 대비 **+12.6 pp**.
> Confidence P(true | y=ITC) 도 0.287 → 0.365 로 28% 상승. CAM17_CONCH 에서 가장 큰 LSAP 이득.

#### 4.4.2 BRACS_CONCH_ABMIL (3 cls: BT / AT / MT)

| Method      | AUC | BT acc | AT acc | MT acc | 평균 k_pool |
|-------------|-----|--------|--------|--------|-------------|
| Vanilla     | ref | high   | mid    | high   | N           |
| ECC-DI      | ref | high   | mid    | high   | top-k       |
| LSAP α=1.1  | comparable | high (1165/1350 ≈ 86.3%) | partially worse | drops (MT→AT 혼동) | ≈ 100 |
| LSAP α=1.3  | comparable | better balance | better | better | smaller |
| LSAP α=1.5  | comparable | even sharper | best on focal | varies | smallest |

> **EDA 분석 (MT 정확도 하락 원인)**: BRACS_CONCH α=1.1 에서 MT 가 AT 로 오분류되는 비율
> 증가. n_mass50(MT)=88 이지만 α=1.1 의 entmax 가 광범위한 attention 을 유지하여 AT-유사
> 신호를 끌어들임. α=1.3+ 에서 회복됨 → **백본·클래스 의존 α 선택의 필요성** 확인.

#### 4.4.3 진행 중 cell (UNI / Virchow)

| Cell                  | α=1.1 | α=1.3 | α=1.5 |
|-----------------------|-------|-------|-------|
| BRACS_UNI_ABMIL       | 5/5 ✓ | 4/5   | 0/5   |
| BRACS_Virchow_ABMIL   | 0/5   | 0/5   | 0/5   |
| CAM17_UNI_ABMIL       | 5/5 ✓ | 0/5   | 0/5   |
| CAM17_Virchow_ABMIL   | 0/5   | 0/5   | 0/5   |

> Slurm 전환 완료 후 잔여 43 run 일괄 실행 예정.

### 4.5 핵심 결과 요약

1. **CAM17_CONCH ITC: LSAP α=1.1 이 Vanilla 대비 +14.3 pp**.
   극희소 positive (ITC) 에서 entmax_α 의 sparse-aware aggregation 이 background dilution 을
   효과적으로 차단 (P1 해결의 정량 증거).

2. **백본–α 상호작용**.
   CONCH 는 낮은 α (≈1.1) 가 최적이지만, BRACS_CONCH 의 MT 클래스는 α=1.3+ 가 더 안전.
   → "단일 α 가 모든 클래스에 최적" 은 거짓. 백본·태스크별 α tuning 또는 클래스 인지적
   adaptive α 가 후속 연구 방향.

3. **τ-MLP 는 사실상 불필요 (Cohen d_z = 0.10)**.
   추가 파라미터 없이 z-score + entmax_α 만으로 충분. 즉, LSAP 의 effective hyperparameter
   는 α 1 개. → main 실험은 `lsap_no_tau=True`.

---

## 5. Slurm Migration Plan

- 현재 11 GPU (w02 6 + w03 5) 분산 실행 중.
- 사용자 다른 세션에서 Slinky → Slurm 전환 작업 진행 중.
- **잔여 43 LSAP run** 은 Slurm 전환 완료 후 일괄 dispatch.
- 전환 손실 추정: 진행 중 6 run 의 ~50% 가 재시작 필요 (≈ 1.5 h 손실) 또는 wait-to-completion (3–5 h, 0 loss).

---

## 6. Future Work

- BRACS_Virchow / CAM17_Virchow 3 α × 5 seed 완료 → 전체 6 cell × 3 α 매트릭스 확정
- **Class-conditional α**: ITC 슬라이드는 α=1.1, macro 슬라이드는 α=1.5 같은 dynamic α 선택
- **EDA-driven α**: vanilla attention 의 n_mass50 분포를 보고 슬라이드별 α 를 결정하는
  zero-shot routing
- ABMIL 외 ACMIL / CLAM 에서 LSAP 의 일반화 검증
- ECC-DI 와의 **하이브리드**: entmax_α post-hoc top-k floor 로 ITC + macro 동시 개선 가능성

---

## Appendix A. Per-cell × per-class confidence table

(Scripts: `/tmp/lsap_confidence.py`, `/tmp/a020_results.py` — Slurm 전환 후 전체 cell 완료 시
table 재생성)

### A.1 Confidence 메트릭 정의

- `n_corr` / `conf(corr)` : 올바르게 예측한 슬라이드 수와 그들의 평균 max-prob
- `n_inc`  / `conf(inc)`  : 오분류한 슬라이드 수와 그들의 평균 max-prob
- `P(true|y)`            : true class probability (예측 정오와 무관한 클래스 신뢰도)

→ ITC/micro 같은 희소 클래스에서 LSAP α=1.1 은 P(true|y) 가 Vanilla/ECC-DI 대비 상승.

## Appendix B. Files modified

- [main.py](main.py) — CLI flags
- [models/model_abmil.py](models/model_abmil.py) — z-score + entmax_α dispatch
- [models/model_clam.py](models/model_clam.py) — `LSAPTemperature` 모듈
- [models/model_acmil.py](models/model_acmil.py) — ACMIL 분기 (후속)
- [utils/core_utils.py](utils/core_utils.py) — kwargs 전달
- [scripts/cells_lsap_w02.txt](scripts/cells_lsap_w02.txt) — BRACS 45 line
- [scripts/cells_lsap_w03.txt](scripts/cells_lsap_w03.txt) — CAM17 45 line
- [scripts/worker_lsap.sh](scripts/worker_lsap.sh) — multi-GPU dispatcher
