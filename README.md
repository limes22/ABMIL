# ABMIL + LSAP

ABMIL (Ilse et al. 2018) reference implementation + **LSAP (Learned Sparse Attention
Pooling)** — softmax aggregation 을 z-score standardization + entmax_α aggregation
으로 교체하는 drop-in replacement.

본 저장소는 CLAM (Lu et al. 2021) 코드를 베이스로 한 ABMIL/ACMIL 학습 파이프라인에
LSAP 를 통합한 작업 코드와 EDA 스크립트, 결과 리포트를 포함한다.

## 문서

- **[docs/LSAP_REPORT.md](docs/LSAP_REPORT.md)** — LSAP 의 문제 정의, EDA, 메소드,
  6 cell × 3 α ablation 결과, Vanilla / ECC-DI 와의 비교.

## 빠른 사용법

```bash
# Vanilla ABMIL
python main.py --task task_X --model_type abmil --attn_norm softmax ...

# LSAP (α=1.1, τ ≡ 1)
python main.py --task task_X --model_type abmil \
    --attn_norm entmax_alpha --lsap_no_tau --lsap_alpha 1.1 ...

# LSAP with learned per-slide temperature (참고용, 본실험에선 미사용)
python main.py --task task_X --model_type abmil \
    --attn_norm entmax_alpha --lsap_temp --lsap_alpha 1.1 ...
```

핵심 CLI 인자 ([main.py](main.py)):

| flag | 값 | 효과 |
|------|-----|------|
| `--attn_norm` | `softmax` / `sparsemax` / `entmax15` / `entmax_alpha` | attention normalization 선택 |
| `--lsap_temp` | bool | learned per-slide τ-MLP 활성 |
| `--lsap_no_tau` | bool | z-score 만 적용, τ ≡ 1 (default LSAP) |
| `--lsap_alpha` | float (1.0~2.0) | entmax_bisect 의 α |
| `--lsap_eps` | float | τ 의 softplus lower bound |

## 디렉토리 구조

```
.
├── main.py                # 학습 entry point
├── models/                # ABMIL / ACMIL / CLAM / ECSA
│   ├── model_abmil.py     # LSAP integration (forward path)
│   ├── model_clam.py      # LSAPTemperature module
│   └── model_acmil.py     # ACMIL (ECC-DI 비교용)
├── utils/
│   └── core_utils.py      # train loop, LSAP kwargs plumbing
├── dataset_modules/       # CLAM dataset interface
├── scripts/
│   ├── cells_lsap*.txt    # cell × α × seed grid
│   └── worker_lsap.sh     # multi-GPU dispatcher (skip-if-done, lock)
├── eda/                   # EDA / 결과 분석 스크립트
│   ├── _eda_lowdim*.py    # n_mass50, vanilla attention 분석
│   ├── _lsap_*.py         # LSAP 결과 집계
│   ├── _method_confidence_all.py  # Vanilla / ECC-DI / LSAP 신뢰도 비교
│   ├── _mt_drop_eda.py    # BRACS_CONCH MT 정확도 하락 원인 분석
│   └── lsap_confidence.py # per-class confidence dump
└── docs/
    └── LSAP_REPORT.md     # 전체 보고서
```

## License

Base code (CLAM): GPLv3 (see [LICENSE.md](LICENSE.md)).
LSAP additions: 동일 라이선스 승계.
