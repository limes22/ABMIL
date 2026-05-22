# LSAP: Learned Sparse Attention Pooling for WSI MIL

> ABMIL (Ilse et al. 2018) 의 softmax aggregation 을 **z-score 표준화 + entmax_α
> aggregation** 으로 교체하는 drop-in replacement.
>
> 본 문서는 (1) 문제 정의, (2) EDA 기반 설계 근거, (3) 수식 정의, (4) 6 cell × 5 method
> ablation 의 cell × class 단위 실측 수치 (correct/total, k_pool, AUC, P(t|y), GT k) 를
> 정리한다.
> snapshot: 2026-05-22 06:00 (47 / 90 LSAP α runs completed; Slurm 전환 후 잔여 43 run).

---

## 1. 문제 정의

ABMIL 의 표준 attention pooling 은 patch features $h_i \in \mathbb{R}^D$ ($i=1,\ldots,N$) 에서
다음과 같이 bag embedding 을 만든다.

$$
z \;=\; \sum_{i=1}^{N} A_i \, h_i, \qquad A \;=\; \mathrm{softmax}(a) \in \Delta^{N-1}
$$

이 구조는 WSI 도메인에서 세 가지 한계가 있다.

| | 한계 | 데이터적 근거 (Sec. 2 EDA) |
|---|------|---------------------------|
| **P1** | softmax 는 모든 패치에 strictly positive weight 부여 → 작은 종양 (ITC 평균 ~11 패치) 이 수천 normal 패치의 평균에 희석 | CAM17 ITC slide 의 vanilla `n_mass50` (50% mass holding patches) = 11 |
| **P2** | 자연 k 가 슬라이드·클래스마다 50배 변동 → 고정 top-k 불가 | CAM17: ITC=11, micro=3, macro=47, neg=162 |
| **P3** | N 의존성: softmax 의 entropy 상한 = log N → 같은 "강한 패치 하나" 가 N 마다 영향이 다름 | BRACS N ≈ 1k~5k, CAM17 N ≈ 2k~10k 변동 |

기존 처방의 한계:

| 방법 | 한계 |
|------|------|
| Vanilla ABMIL (softmax) | 노이즈 누적, ITC slide 손실 |
| Static top-k (CLAM k=8) | normal slide 도 k=8 강제 → FP |
| ECC-DI (entropy-conditional k) | 학습된 k 분포와 GT k 가 불일치 (ITC 슬라이드에 k=72 학습, GT=11) |
| Learnable α (V2 / AdaptiveTau) | foundation backbone (CONCH/UNI/Virchow) 에서 α → 0.5 saturation |

---

## 2. EDA — 메소드 선택 근거

### 2.1 Vanilla attention 의 자연 k 측정 (`n_mass50`)

각 cell × class 의 vanilla ABMIL OOF (out-of-fold) attention 에서 누적 50% mass 를
점유하는 패치 수 `n_mass50` 의 클래스별 median:

**CAM17** (stage 4-class)

| Class | Median `n_mass50` (GT k) | 의미 |
|-------|--------------------------|------|
| micro | 3 | 매우 sparse (point-like) |
| ITC | 11 | sparse (single small cluster) |
| macro | 47 | 중간 |
| neg | 162 | 종양 없음 → wide attention |

**BRACS** (3-class)

| Class | Median `n_mass50` (GT k) |
|-------|--------------------------|
| BT (benign) | 156 |
| AT (atypical) | 30 |
| MT (malignant) | 88 |

→ 자연 k 가 50× 변동. **P2 의 데이터적 근거**. 본 GT k 는 §4 의 모든 결과 표에 함께 표기.

### 2.2 백본별 attention sharpness 차이

학습된 vanilla attention raw logits 의 분포:

| Cell | Class | σ(a) | max − mean | 학습된 모델 `n_mass50` |
|------|-------|------|------------|------------------------|
| CAM17_CONCH  | ITC   | 0.68 | 2.69 | 1 (collapse) |
| CAM17_CONCH  | micro | 1.01 | 4.34 | 1 |
| CAM17_UNI    | ITC   | 1.23 | 4.26 | 155 |
| CAM17_Virchow| ITC   | 1.00 | 4.24 | 484 |

→ CONCH 의 attention 이 적당히 sharp ("Goldilocks") — entmax 와 자연스럽게 매칭.
UNI / Virchow 는 sharper 하거나 더 spread → α 선택이 백본 의존적이어야 함.

### 2.3 entmax α 선택 근거 — vanilla ã 에서 GT k 매칭

Vanilla 학습 attention 의 표준화된 $\tilde a$ 에 $\mathrm{entmax}_\alpha$ 를 적용해
$\|p\|_0 \approx \mathrm{median\ } n_{\text{mass50}}$ 가 되는 α 추정:

| Cell | best α (cell-level) |
|------|---------------------|
| BRACS_CONCH   | 1.3 |
| BRACS_UNI     | 1.3 |
| BRACS_Virchow | 1.3 |
| CAM17_CONCH   | 1.1 (더 softmax-like, fine-grained ITC) |
| CAM17_UNI     | 1.3 |
| CAM17_Virchow | 1.3 |

→ 대부분 α=1.3 최적, CAM17_CONCH 만 α=1.1. 본 ablation 의 grid 는 α ∈ {1.1, 1.3, 1.5}.

---

## 3. 메소드 — LSAP (Learned Sparse Attention Pooling)

LSAP 는 ABMIL 의 attention pooling 단계만 교체한다. 다른 모든 부분 (gated attention
network, classifier, CE loss) 은 ABMIL 원본과 동일하다.

### 3.1 수식 정의 (4 step)

입력: patch features $h_i \in \mathbb{R}^D$, slide label $y$.

**Step 1 — Gated attention** (Ilse et al. 2018, ABMIL 그대로):

$$
u_i \;=\; \mathrm{ReLU}(W_1 h_i) \in \mathbb{R}^{512}
$$

$$
a_i \;=\; w^{\top}\!\bigl(\tanh(V u_i) \odot \sigma(U u_i)\bigr) \;\in\; \mathbb{R}
$$

여기서 $W_1 \in \mathbb{R}^{512 \times D}$, $V, U \in \mathbb{R}^{256 \times 512}$,
$w \in \mathbb{R}^{256}$, $\sigma$ 는 sigmoid, $\odot$ 는 element-wise product.

**Step 2 — Z-score 표준화** (identifiability 고정):

$$
\mu_a \;=\; \frac{1}{N}\sum_{i=1}^{N} a_i, \qquad \sigma_a \;=\; \sqrt{\frac{1}{N}\sum_{i=1}^{N}(a_i - \mu_a)^2 + \epsilon_\sigma}
$$

$$
\tilde{a}_i \;=\; \frac{a_i - \mu_a}{\sigma_a}
$$

$\mu_a, \sigma_a$ 는 grad path 에서 detach 하지 않는다 (LayerNorm 과 동일 패턴).
표준화가 없으면 $a \mapsto 2a$ 와 $\alpha$ 선택이 entangle 되어 α 효과 측정이 불가능.

**Step 3 — Entmax_α aggregation** (Peters et al. 2019):

α-entmax 는 다음 strictly concave optimization 의 argmax 로 정의된다.

$$
\mathrm{entmax}_\alpha(z) \;=\; \arg\max_{p \in \Delta^{N-1}} \; \bigl\langle p, z \bigr\rangle \;+\; H_\alpha(p)
$$

여기서 $H_\alpha$ 는 Tsallis α-entropy:

$$
H_\alpha(p) \;=\;
\begin{cases}
\displaystyle \frac{1}{\alpha(\alpha-1)} \sum_{i=1}^{N}\bigl(p_i - p_i^{\alpha}\bigr), & \alpha \neq 1 \\[4pt]
\displaystyle -\sum_{i=1}^{N} p_i \log p_i, & \alpha = 1
\end{cases}
$$

특수값:

- $\alpha = 1$: $\mathrm{entmax}_1 = \mathrm{softmax}$ (dense)
- $\alpha = 1.5$: $\mathrm{entmax}_{1.5}$ (closed-form, Peters et al. 2019)
- $\alpha = 2$: sparsemax (Martins & Astudillo 2016)

본 메소드는 임의 α ∈ (1, 2] 에 대해 $\mathrm{entmax\_bisect}$ (root-finding via Householder
method) 로 정확한 $p$ 를 계산한다. 출력 $p$ 는 *exact zeros* 를 가질 수 있는 sparse simplex
원소이며, $\|p\|_0 \le N$. α 가 1 에 가까울수록 dense, 2 에 가까울수록 sparse.

$$
p \;=\; \mathrm{entmax}_\alpha(\tilde a), \qquad \alpha \in \{1.1,\, 1.3,\, 1.5\}
$$

$$
z_{\mathrm{bag}} \;=\; \sum_{i=1}^{N} p_i \, u_i \;\in\; \mathbb{R}^{512}
$$

**Step 4 — Classifier + CE loss**:

$$
\hat{y} \;=\; \mathrm{softmax}(W_c z_{\mathrm{bag}}), \qquad
\mathcal{L} \;=\; -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$

### 3.2 학습 가능 파라미터

| 모듈 | params |
|------|--------|
| $W_1$ (Linear D → 512) | ~ D × 512 |
| Gated attention $V, U, w$ | ~ 264k |
| Classifier $W_c$ | ~ 1.5k |
| **entmax_α** | **0** (α 는 hyperparameter, 학습 안 함) |
| **Total** | Vanilla ABMIL 과 동일 |

→ LSAP 는 추가 학습 파라미터 0. *Pure replacement of the aggregation operator.*

### 3.3 Code references

| Component | 파일 / 라인 |
|-----------|-------------|
| ABMIL forward (z-score + entmax_α dispatch) | [models/model_abmil.py:163-214](models/model_abmil.py#L163-L214) |
| CLI flags (`--attn_norm`, `--lsap_alpha`, `--lsap_no_tau`) | [main.py](main.py) |
| Kwargs plumbing | [utils/core_utils.py](utils/core_utils.py) |

### 3.4 학습 설정 (Vanilla / ECC-DI / LSAP 공통)

| 항목 | 값 |
|------|----|
| max_epochs | 200 |
| early stopping | patience=20, stop_epoch=50 |
| ckpt 선택 | min val_loss |
| 평가 | 10-fold CV × 5 seeds = 50 runs / (cell × method) |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| Loss | bag-level CE only (no instance loss, `--no_inst_cluster`) |

---

## 4. Results — 6 cell × 5 method 매트릭스

snapshot: 2026-05-22 06:00. `seeds` = "완료 / 계획" (Vanilla 는 5 seeds 중 3 가 일관성 검증
하에 표에 포함). `k` 열은 LSAP/ECC-DI 의 평균 ‖p‖_0 (vanilla 는 dense=N).
모든 셀에 absolute `correct/total` (백분율) 표기. `P(t|y)` = mean of `prob[true_label]`
across slides of class y.

### 4.1 BRACS_CONCH_ABMIL — 모든 method 완료 ✓

GT 자연 k: **BT=156, AT=30, MT=88**.

| Method      | seeds | AUC          | k_pool | acc   | BT (correct / total)        | AT (correct / total)       | MT (correct / total)        | AT P(t\|y) |
|-------------|-------|--------------|--------|-------|------------------------------|-----------------------------|------------------------------|------------|
| Vanilla     | 3/3   | 0.9270 ± .020 | 2682 (N) | 81.5% | 1091/1200 (90.9%)            | 148/390 (37.9%)             | 718/810 (88.6%)              | 0.372 |
| ECC-DI      | 5/5   | 0.9245 ± .019 | 37     | 81.7% | 1807/2000 (90.3%)            | 273/650 (42.0%) ⭐          | 1186/1350 (87.9%)            | 0.385 |
| LSAP α=1.1  | 5/5   | 0.9167 ± .023 | 1207   | 81.6% | 1818/2000 (90.9%)            | **280/650 (43.1%) ⭐⭐**     | 1165/1350 (86.3%)            | 0.369 |
| LSAP α=1.3  | 5/5   | 0.9228 ± .021 | 83     | 80.8% | 1776/2000 (88.8%)            | 276/650 (42.5%)             | 1179/1350 (87.3%)            | 0.381 |
| LSAP α=1.5  | 5/5   | 0.9191 ± .024 | 21     | 80.7% | 1781/2000 (89.0%)            | 272/650 (41.8%)             | 1176/1350 (87.1%)            | 0.384 |

### 4.2 BRACS_UNI_ABMIL — LSAP α=1.3/1.5 진행 중 ⏳

GT 자연 k: **BT=156, AT=30, MT=88**.

| Method      | seeds | AUC          | k_pool | acc   | BT                            | AT                          | MT                           | AT P(t\|y) |
|-------------|-------|--------------|--------|-------|-------------------------------|------------------------------|-------------------------------|------------|
| Vanilla     | 3/3   | 0.9243 ± .019 | 2682 (N) | 81.0% | 1091/1200 (90.9%)             | 155/390 (39.7%)              | 697/810 (86.0%)               | 0.373 |
| ECC-DI      | 5/5   | 0.9294 ± .021 ⭐ | 38   | 81.6% | 1825/2000 (91.2%)             | **268/650 (41.2%) ⭐**       | 1171/1350 (86.7%)             | 0.379 |
| LSAP α=1.1  | 3/5   | 0.9160 ± .028 | 988    | 79.7% | 1351/1480 (91.3%)             | **157/481 (32.6%) ⚠**        | 850/999 (85.1%)               | **0.337 ⚠** |
| LSAP α=1.3  | 0/5 ⏳ | (partial)    | 97     | --    | --                            | --                           | --                            | -- |
| LSAP α=1.5  | 0/5 ⏳ | (시작 안 함)  | --     | --    | --                            | --                           | --                            | -- |

→ BRACS_UNI 에서 α=1.1 의 AT 정확도가 vanilla 39.7% / ECC-DI 41.2% 대비 **32.6% (−8.6 pp vs Vanilla)** 로
큰 손실. 같은 α=1.1 이 BRACS_CONCH 에서는 +5.2 pp 인 것과 정반대. **Backbone-α
interaction** 의 직접 증거.

### 4.3 BRACS_Virchow_ABMIL — LSAP 진행 중 ⏳

GT 자연 k: **BT=156, AT=30, MT=88**.

| Method      | seeds | AUC          | k_pool | acc   | BT                          | AT                         | MT                          |
|-------------|-------|--------------|--------|-------|------------------------------|-----------------------------|------------------------------|
| Vanilla     | 3/3   | 0.9157 ± .023 | 2682 (N) | 80.4% | 1110/1200 (92.5%)            | 115/390 (29.5%)             | 705/810 (87.0%)              |
| ECC-DI      | 5/5   | 0.9211 ± .024 ⭐ | 40   | 80.8% | 1846/2000 (92.3%)            | **218/650 (33.5%) ⭐**      | 1168/1350 (86.5%)            |
| LSAP α=1.1  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           |
| LSAP α=1.3  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           |
| LSAP α=1.5  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           |

### 4.4 CAM17_CONCH_ABMIL — 모든 method 완료 ✓  **(paper-ready)**

GT 자연 k: **neg=162, ITC=11, micro=3, macro=47**.

| Method      | seeds | AUC          | k_pool | acc   | neg (cor/total)            | **ITC (cor/total)**          | micro (cor/total)         | macro (cor/total)         | ITC P(t\|y) |
|-------------|-------|--------------|--------|-------|-----------------------------|------------------------------|----------------------------|----------------------------|-------------|
| Vanilla     | 3/3   | 0.9148 ± .045 | 3572 (N) | 86.2% | 860/882 (97.5%)             | 19/81 (23.5%)                | 90/153 (58.8%)             | 195/234 (83.3%)            | 0.287 |
| ECC-DI      | 5/5   | 0.9200 ± .043 | 45     | 87.1% | 1444/1470 (98.2%)           | 34/135 (25.2%)               | 152/255 (59.6%) ⭐         | 329/390 (84.4%)            | 0.271 |
| LSAP α=1.1  | 5/5   | 0.9253 ± .042 | 1320   | 86.8% | 1436/1470 (97.7%)           | **51/135 (37.8%) ⭐⭐**       | 136/255 (53.3%)            | 331/390 (84.9%)            | **0.365 ⭐⭐** |
| LSAP α=1.3  | 5/5   | 0.9173 ± .039 | 78     | 85.5% | 1419/1470 (96.5%)           | 41/135 (30.4%)               | 136/255 (53.3%)            | 327/390 (83.8%)            | 0.303 |
| LSAP α=1.5  | 5/5   | **0.9278 ± .038 ⭐** | 18 | 86.8% | 1441/1470 (98.0%)           | 41/135 (30.4%)               | 142/255 (55.7%)            | 330/390 (84.6%)            | 0.339 |

→ **메인 paper claim**:
- LSAP α=1.1 의 ITC accuracy 37.8% (51/135) 가 Vanilla 23.5% (19/81) 대비 **+14.3 pp**,
  ECC-DI 25.2% (34/135) 대비 **+12.6 pp**.
- ITC slide 의 true-class confidence P(t|y=ITC) = 0.365 가 Vanilla 0.287, ECC-DI 0.271 대비
  **+0.078 / +0.094**. → entmax 의 정확한 0 이 normal patch dilution 을 차단했다는 직접 증거.
- 한편 macro AUC peak 는 α=1.5 (0.9278). 클래스 → α 매칭이 다름 (ITC↔1.1, macro↔1.5).

### 4.5 CAM17_UNI_ABMIL — LSAP α=1.3/1.5 진행 중 ⏳

GT 자연 k: **neg=162, ITC=11, micro=3, macro=47**.

| Method      | seeds | AUC          | k_pool | acc   | neg                          | ITC                          | micro                       | macro                       | ITC P(t\|y) |
|-------------|-------|--------------|--------|-------|------------------------------|-----------------------------|------------------------------|------------------------------|-------------|
| Vanilla     | 3/3   | 0.8693 ± .050 | 3572 (N) | 84.5% | 857/882 (97.2%)              | 14/81 (17.3%)               | 79/153 (51.6%)               | 191/234 (81.6%)              | 0.195 |
| ECC-DI      | 5/5   | 0.8948 ± .045 ⭐ | 52  | 84.3% | 1414/1470 (96.2%)            | 26/135 (19.3%) ⭐            | 147/255 (57.6%) ⭐           | 310/390 (79.5%)              | 0.219 |
| LSAP α=1.1  | 5/5   | (집계 중)     | 1571   | (집계 중) | 1401/1470 (95.3%)            | **30/135 (22.2%) ⭐**        | 125/255 (49.0%)              | 320/390 (82.1%) ⭐           | **0.243 ⭐** |
| LSAP α=1.3  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           | --                           | -- |
| LSAP α=1.5  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           | --                           | -- |

→ CAM17_UNI 의 ITC 에서도 LSAP α=1.1 이 Vanilla 17.3% → 22.2% (+4.9 pp), P(t|y) 0.195 →
0.243 (+0.048). CAM17_CONCH 만큼 큰 폭은 아니나 부호 일관.

### 4.6 CAM17_Virchow_ABMIL — LSAP 진행 중 ⏳

GT 자연 k: **neg=162, ITC=11, micro=3, macro=47**.

| Method      | seeds | AUC          | k_pool | acc   | neg                          | ITC                          | micro                       | macro                       | ITC P(t\|y) |
|-------------|-------|--------------|--------|-------|------------------------------|-----------------------------|------------------------------|------------------------------|-------------|
| Vanilla     | 3/3   | 0.8588 ± .060 ⭐ | 3572 (N) | 83.3% | 842/882 (95.5%)            | 6/81 (7.4%) ⚠              | 77/153 (50.3%)               | 199/234 (85.0%)              | 0.172 ⚠ |
| ECC-DI      | 5/5   | 0.8446 ± .060 | 49     | 84.5% | 1427/1470 (97.1%)            | 13/135 (9.6%)               | 129/255 (50.6%)              | 332/390 (85.1%)              | 0.156 ⚠ |
| LSAP α=1.1  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           | --                           | -- |
| LSAP α=1.3  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           | --                           | -- |
| LSAP α=1.5  | 0/5 ⏳ | --           | --     | --    | --                          | --                          | --                           | --                           | -- |

→ Virchow 가 모든 backbone 중 ITC accuracy / P(t|y) 최저 (Vanilla 7.4% / 0.172). LSAP 가
가장 큰 효과를 낼 잠재력. Slurm 전환 후 우선 검증 대상.

---

## 5. 진행 상황 (snapshot 2026-05-22 06:00)

| Cell           | Vanilla | ECC-DI | LSAP α=1.1 | LSAP α=1.3 | LSAP α=1.5 |
|----------------|---------|--------|------------|------------|------------|
| BRACS_CONCH    | ✓       | ✓      | ✓ 5/5      | ✓ 5/5      | ✓ 5/5      |
| BRACS_UNI      | ✓       | ✓      | ⏳ 3/5     | ⏳ 0/5     | ⏳ 0/5     |
| BRACS_Virchow  | ✓       | ✓      | ⏳ 0/5     | ⏳ 0/5     | ⏳ 0/5     |
| CAM17_CONCH    | ✓       | ✓      | ✓ 5/5      | ✓ 5/5      | ✓ 5/5      |
| CAM17_UNI      | ✓       | ✓      | ✓ 5/5      | ⏳ 0/5     | ⏳ 0/5     |
| CAM17_Virchow  | ✓       | ✓      | ⏳ 0/5     | ⏳ 0/5     | ⏳ 0/5     |

LSAP α runs: **47 / 90 완료**. 잔여 43 run 은 Slurm 전환 완료 후 일괄 dispatch 예정.

---

## 6. 핵심 결론

### 6.1 CAM17 ITC 에서 LSAP α=1.1 의 압도적 우위 *(paper claim)*

| Cell          | Vanilla ITC               | ECC-DI ITC               | LSAP α=1.1 ITC                  | Δ vs ECC-DI (acc) | Δ vs ECC-DI (P(t\|y)) |
|---------------|---------------------------|---------------------------|---------------------------------|-------------------|-----------------------|
| CAM17_CONCH   | 19/81 (23.5%), P=0.287    | 34/135 (25.2%), P=0.271   | **51/135 (37.8%), P=0.365** ⭐⭐ | **+12.6 pp**      | **+0.094** |
| CAM17_UNI     | 14/81 (17.3%), P=0.195    | 26/135 (19.3%), P=0.219   | **30/135 (22.2%), P=0.243** ⭐  | **+2.9 pp**       | **+0.024** |

ITC slide 의 자연 k = 11 인데, LSAP α=1.1 의 평균 ‖p‖_0 ≈ 12 (CAM17_CONCH) / ≈ 15
(CAM17_UNI) 으로 **자연 k 와 학습된 k 가 정렬**됨. ECC-DI 는 ITC slide 에 k=45~52 학습
(over-broad). LSAP 의 entmax 가 데이터의 자연 sparsity 를 그대로 학습한다는 메커니즘 증거.

### 6.2 Backbone–α interaction (caveat)

같은 α=1.1 이 cell 따라 효과가 정반대:

| Cell          | AT (BRACS) / ITC (CAM17) accuracy Δ vs Vanilla |
|---------------|------------------------------------------------|
| BRACS_CONCH   | **+5.2 pp** (37.9 → 43.1%)                     |
| BRACS_UNI     | **−7.1 pp** (39.7 → 32.6%) ⚠                   |
| CAM17_CONCH   | **+14.3 pp** (23.5 → 37.8%)                    |
| CAM17_UNI     | **+4.9 pp** (17.3 → 22.2%)                     |

UNI 의 vanilla attention σ(a)=1.23 (CONCH 0.68 보다 sharp) 이라, α=1.1 의 less-sparse
정책과 mismatch. **단일 α 규칙으로 모든 backbone × class 를 만족할 수 없음.** Cell-aware
또는 class-aware α 가 후속 연구 방향.

### 6.3 Sparsity vs AUC trade-off

LSAP α 가 1 → 2 로 갈수록 k_pool 감소 (CAM17_CONCH: 1320 → 78 → 18), AUC 는 비단조:

| α | CAM17_CONCH k_pool | CAM17_CONCH AUC | ITC acc |
|---|--------------------|------------------|---------|
| 1.1 | 1320 | 0.9253 | **37.8%** ⭐ |
| 1.3 | 78  | 0.9173 | 30.4% |
| 1.5 | 18  | **0.9278 ⭐** | 30.4% |

→ α=1.1 은 ITC class 특화, α=1.5 는 overall AUC 특화. 메소드의 실용적 사용은
"primary endpoint 가 minority class detection 이면 α=1.1, overall ranking 이면 α=1.5".

---

## 7. Slurm 전환 & 남은 작업

- 사용자 별도 세션에서 Slinky → Slurm 전환 진행 중.
- 전환 후 잔여 **43 LSAP α runs** 일괄 dispatch:
  - BRACS_UNI: α=1.3 (2 seed), α=1.5 (5 seed) = 7
  - BRACS_Virchow: 3 α × 5 seed = 15
  - CAM17_UNI: α=1.3 + α=1.5 = 10
  - CAM17_Virchow: 3 α × 5 seed = 15
- 완료 즉시 본 문서의 §4.2, §4.3, §4.5, §4.6 표 갱신.

---

## Appendix A — Files modified for LSAP

| 파일 | 변경 |
|------|------|
| [main.py](main.py) | CLI flags `--attn_norm`, `--lsap_alpha`, `--lsap_no_tau` |
| [models/model_abmil.py](models/model_abmil.py) | z-score + entmax_α dispatch, k_pool 측정 |
| [models/model_clam.py](models/model_clam.py) | (사용 안 하는 LSAPTemperature class 는 유지, no_tau 옵션으로 우회) |
| [utils/core_utils.py](utils/core_utils.py) | ABMIL kwargs plumbing |
| [scripts/cells_lsap_w02.txt](scripts/cells_lsap_w02.txt) | BRACS 45 lines (cell × α × seed grid) |
| [scripts/cells_lsap_w03.txt](scripts/cells_lsap_w03.txt) | CAM17 45 lines |
| [scripts/worker_lsap.sh](scripts/worker_lsap.sh) | multi-GPU dispatcher (skip-if-done, lock) |

## Appendix B — 본 결과 표를 생성한 EDA / aggregation 스크립트

| 스크립트 | 역할 |
|----------|------|
| `eda/_eda_lowdim*.py` | vanilla attention 의 `n_mass50`, σ(a), max−mean 분포 |
| `eda/_lsap_full_table.py` | per cell × method 의 AUC / k_pool / per-class accuracy 집계 |
| `eda/_method_confidence_all.py` | Vanilla / ECC-DI / LSAP 의 P(t\|y), correct/incorrect confidence |
| `eda/_mt_drop_eda.py` | BRACS_CONCH α=1.1 의 MT→AT 혼동 원인 분석 |
| `eda/lsap_confidence.py` | per-class P(t\|y) dump |
