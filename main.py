from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'abmil', 'abmil_official', 'acmil', 'dsmil', 'transmil', 'dtfd'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_camelyon16', 'task_tcga_nsclc', 'task_camelyon17', 'task_bracs'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
### Dynamic k options
parser.add_argument('--dynamic_k', action='store_true', default=False,
                     help='use entropy-based dynamic k for instance clustering')
parser.add_argument('--k_min', type=int, default=4,
                     help='minimum k for dynamic k sampling (default: 4)')
parser.add_argument('--k_max', type=int, default=16,
                     help='maximum k for dynamic k sampling (default: 16)')
parser.add_argument('--gamma', type=float, default=1.0,
                     help='gamma for nonlinear entropy-to-k mapping (default: 1.0, >1 favors smaller k)')
parser.add_argument('--dk_method', type=str, default='v2', choices=['v1', 'v2', 'v3'],
                     help='dynamic k method: v1 (basic), v2 (top-M, patch bias removed), v3 (entropy+confidence)')
parser.add_argument('--dk_inverse', action='store_true', default=False,
                     help='inverse dynamic-k: sharp attention → large k (explore more when confident)')
### Adaptive K Range (패치 수 비례)
parser.add_argument('--adaptive_k_range', action='store_true', default=False,
                     help='set k_min/k_max proportional to patch count N (ignores --k_min/--k_max)')
parser.add_argument('--k_min_pct', type=float, default=0.001,
                     help='adaptive k_min = max(4, N * k_min_pct) (default: 0.001 = 0.1%%)')
parser.add_argument('--k_max_pct', type=float, default=0.01,
                     help='adaptive k_max = max(32, N * k_max_pct) (default: 0.01 = 1%%)')
### ECC-DI: parameterize the hardcoded floors on (k_min, k_max). Standard ECSA uses (4, 32).
### ECC-DI spec sets k_min_floor=8, k_max_floor=8 (no 32 floor; k_max = max(8, N*k_max_pct)).
parser.add_argument('--k_min_floor', type=int, default=None,
                     help='override the k_min floor (default None=4). ECC-DI uses 8.')
parser.add_argument('--k_max_floor', type=int, default=None,
                     help='override the k_max floor (default None=32). ECC-DI uses 8.')
parser.add_argument('--feature_extractor', type=str, default='resnet50',
                     choices=['resnet50', 'ctranspath', 'uni', 'virchow', 'conch'],
                     help='feature extractor used (determines embed_dim and data_dir suffix)')
### Uncertainty Loss Weighting (Kendall et al., CVPR 2018)
parser.add_argument('--uncertainty_weight', action='store_true', default=False,
                     help='use learnable uncertainty-based loss weighting instead of fixed bag_weight')
### Learnable Attention Temperature
parser.add_argument('--learnable_temp', action='store_true', default=False,
                     help='use learnable global temperature for attention softmax')
parser.add_argument('--adaptive_temp', action='store_true', default=False,
                     help='use per-slide adaptive temperature (overrides --learnable_temp)')
### Sparse Top-K Attention Pooling
parser.add_argument('--sparse_topk', action='store_true', default=False,
                     help='use top-k sparse attention for bag aggregation (requires --dynamic_k)')
### AdaptiveSparsePooling (Phase 1, Feature_Adaptive 통합)
parser.add_argument('--adaptive_sparse_pool', action='store_true', default=False,
                     help='use gated dense+sparse pool (alpha learned from feature stats; '
                          'sparse = top-1%% of N). Helps when initial attention is too flat.')
### Feature-Adaptive (Phase 2 — meta-flag: norm + activation + sparse_pool + temp 모두 활성)
parser.add_argument('--feature_adaptive', action='store_true', default=False,
                     help='Phase 2 meta-flag: enable AdaptiveNorm + AdaptiveActivation + '
                          'AdaptiveSparsePooling + AdaptiveTemperature (all 5 modules). '
                          'Backbone-agnostic plug-in.')
### Bag-size-aware k_max cap (Phase 3 — paper finding: large bags 에서 ECSA 부작용)
parser.add_argument('--k_max_cap', type=int, default=0,
                     help='Cap k_max for AdaptiveSparsePooling regardless of N (default 0 = no cap). '
                          'Useful for very large bags (N>100K) where top-1% still includes too much noise. '
                          'Recommended: 500 for ResNet/CTrans on CAM16.')
parser.add_argument('--entropy_only', action='store_true', default=False,
                     help='Pure-perplexity k = round(exp(H(A))). Bypasses k_min/k_max/k_max_cap. '
                          'Hyperparameter-free, bag-size-independent. Requires --feature_adaptive or --adaptive_sparse_pool.')
parser.add_argument('--entropy_k_floor', type=int, default=8,
                     help='Minimum k when --entropy_only (default 8). Prevents k=1 collapse on sharp-attention models.')
### Log-linear k formula (geometric interpolation, N-independent)
parser.add_argument('--loglinear', action='store_true', default=False,
                     help='Use log-linear k = k_min^(1-H_norm) * k_cap^H_norm (geometric, bounded, N-independent). '
                          'Mutually exclusive with --entropy_only. Requires --feature_adaptive or --adaptive_sparse_pool.')
parser.add_argument('--loglinear_k_min', type=int, default=8,
                     help='Lower bound for k in log-linear mode (default 8).')
parser.add_argument('--loglinear_k_cap', type=int, default=500,
                     help='Upper bound for k in log-linear mode (default 500).')
parser.add_argument('--loglinear_cap_frac', type=float, default=0.0,
                     help='If >0, use N-adaptive cap: k_cap_eff = min(k_cap, cap_frac*N). Helps small-bag datasets like CAM17. cap_frac = k_cap / Q75(N) gives data-driven calibration.')
# Deprecated alias for backward compatibility (existing launch scripts use --loglinear_alpha).
# Same default 0.0 to avoid overriding the new flag's default with None.
parser.add_argument('--loglinear_alpha', type=float, default=0.0, dest='loglinear_cap_frac',
                     help='[Deprecated alias for --loglinear_cap_frac]')
### Phase 1 ablation knobs (ece-v3)
parser.add_argument('--inverse_threshold', type=float, default=1.0,
                     help='Adaptive inverse: only flip H_use=(1-H_norm) when H_norm < threshold. '
                          'Default 1.0 = always invert (legacy). Set to 0.3 for adaptive (only sharp slides). '
                          'Used when --dk_inverse is on.')
parser.add_argument('--hybrid_floor', action='store_true', default=False,
                     help='Hybrid-floor mode: k = min(max(floor, exp(H)), alpha*N). Combines floor8 perplexity with N-adaptive cap.')
parser.add_argument('--hybrid_floor_alpha', type=float, default=0.0,
                     help='Cap fraction for hybrid_floor (e.g. 0.05 → cap at 5% of N). 0 = no cap.')
parser.add_argument('--hybrid_floor_min', type=int, default=8,
                     help='Floor for hybrid_floor (default 8, same as floor8 entropy_only).')
parser.add_argument('--blend_w_bias', type=float, default=0.0,
                     help='Pre-sigmoid bias init for sparse_gate. >0 shifts blend_w toward sparse at start. Recommended: 1.0 (sigmoid≈0.73).')
parser.add_argument('--pure_cap', action='store_true', default=False,
                     help='Pure cap mode: k = alpha*N, no entropy. Ablation to test "is entropy needed at all?".')
parser.add_argument('--pure_cap_frac', type=float, default=0.0,
                     help='Cap fraction for pure_cap mode. e.g. 0.03 for 3% k/N.')
parser.add_argument('--pure_cap_min', type=int, default=8,
                     help='Floor for pure_cap (default 8, same as floor8).')
parser.add_argument('--learnable_alpha', action='store_true', default=False,
                     help='Stage 1 Learnable alpha: cap fraction = sigmoid(nn.Parameter). '
                          'Differentiable soft top-k (sigmoid step at rank=alpha*N). '
                          'Eliminates per-cell alpha sweep — model learns optimal cap from data.')
parser.add_argument('--learnable_alpha_init', type=float, default=0.03,
                     help='Initial sigmoid value for learnable alpha (e.g. 0.03 = start at 3%% cap).')
parser.add_argument('--learnable_alpha_temp', type=float, default=20.0,
                     help='Sigmoid sharpness for soft top-k. Higher = sharper (closer to hard top-k). '
                          'Default 20 → ~0.95 mass within ±0.15 ranks of alpha*N boundary.')
parser.add_argument('--learnable_alpha_min', type=int, default=8,
                     help='Effective floor for learnable alpha (clamps alpha so alpha*N >= this floor).')
parser.add_argument('--learnable_alpha_hybrid', action='store_true', default=False,
                     help='LRA-Hybrid mode: k = min(max(floor, exp(H_use*log(N))), alpha*N) with adaptive inverse on H_use. '
                          'Same as hybrid_floor + adaptive inverse, but alpha is learnable. Use with --dk_inverse --inverse_threshold 0.3.')
### Adaptive τ (cell-specific quantile-based inverse threshold) — ece-v5
parser.add_argument('--adaptive_tau', action='store_true', default=False,
                     help='Adaptive τ: during the warmup epoch run with inverse OFF and collect per-slide H_norm, '
                          'then set inverse_threshold = quantile(H_norm, --adaptive_tau_target). '
                          'Makes sharp_ratio consistent across cells without per-cell sweep of τ.')
parser.add_argument('--adaptive_tau_target', type=float, default=0.3,
                     help='Target quantile of H_norm distribution for adaptive τ (default 0.3 → 30%% slides flip).')
parser.add_argument('--adaptive_tau_warmup_epochs', type=int, default=1,
                     help='Number of epochs in each collection window (default 1).')
parser.add_argument('--adaptive_tau_warmup_start', type=int, default=0,
                     help='First epoch to begin collecting H_norm. Default 0 = collect from the very first '
                          'epoch (untrained model — produces too-high τ that collapses to always-invert by '
                          'mid-training). Recommended 5-10 so that mean_H_norm has stabilized to its '
                          'trained-regime value before the 30%% quantile is taken. '
                          'When >0, the initial inverse_threshold value (e.g. 0.3) is used during pre-warmup epochs.')
parser.add_argument('--adaptive_tau_update_interval', type=int, default=0,
                     help='If >0, re-compute τ every this many epochs after the first finalization (each '
                          're-update opens a fresh warmup_epochs-long collection window without forcing '
                          'inverse OFF). Default 0 = compute τ once and freeze. Recommended 10 to track '
                          'H_norm drift.')
parser.add_argument('--learnable_alpha_lr', type=float, default=0.0,
                     help='Separate learning rate for alpha_logit only (default 0 = use args.lr). '
                          'Set to e.g. 1e-2 (100x base lr) so the 1D scalar gets enough gradient to actually move. '
                          'Without this, alpha typically stays glued to its init value.')
### SOTA plug-in (DSMIL/DTFD-MIL only): add our pooling on top of their architecture
parser.add_argument('--plugin_method', type=str, default='none',
                     choices=['none', 'floor8', 'inverse', 'featad'],
                     help='Plug our entropy-driven k-selection into DSMIL/DTFD-MIL bag-level aggregation.')
parser.add_argument('--plugin_kc', type=int, default=500,
                     help='k_max_cap for plugin (used when plugin_method=inverse|featad).')
### Attention normalization (Phase 2 baseline 비교용)
parser.add_argument('--attn_norm', type=str, default='softmax',
                     choices=['softmax', 'sparsemax', 'entmax15', 'entmax_alpha'],
                     help='attention normalization function (softmax | sparsemax | entmax15 | entmax_alpha). '
                          'entmax_alpha uses --lsap_alpha value via entmax_bisect.')
### Early-stopping ablation: val_AUC-best ckpt 도 같이 저장/평가 (main result 는 val_loss-best 유지)
parser.add_argument('--track_valauc', action='store_true', default=False,
                     help='In addition to val_loss-best checkpoint (default), save the val_AUC-best '
                          'checkpoint as s_X_checkpoint_valauc.pt and report its test AUC alongside.')
### LSAP — Learned Sparse Attention Pooling (entmax15 + per-slide learned τ)
parser.add_argument('--lsap_temp', action='store_true', default=False,
                     help='Use LSAPTemperature (τ-MLP with [mean(a), std(a), max(a), log N] input, '
                          'τ = softplus(MLP) + ε) instead of AdaptiveTemperature. '
                          'Requires --feature_adaptive (or another flag that turns on temp_module). '
                          'For full LSAP method, combine with --attn_norm entmax15.')
parser.add_argument('--lsap_eps', type=float, default=0.01,
                     help='Floor ε for LSAP τ = softplus(MLP)+ε. Smaller ε allows sharper attention. '
                          'Default 0.01 (vs the standalone script which used 0.5).')
parser.add_argument('--lsap_no_tau', action='store_true', default=False,
                     help='LSAP without τ-MLP: standardize a then entmax (τ=1 fixed). '
                          'Used for ablation: "is τ-MLP necessary?"')
parser.add_argument('--lsap_alpha', type=float, default=1.5,
                     help='entmax α value for --attn_norm=entmax_alpha. '
                          '1.0≈softmax, 1.5=entmax15, 2.0=sparsemax.')
### GELU activation (CTransPath 등 zero-centered features 호환성)
parser.add_argument('--use_gelu', action='store_true', default=False,
                     help='Replace ReLU with GELU in feature reduction layer. '
                          'CTransPath features 는 negative 값이 많아 ReLU 가 정보 손실 → GELU 권장.')
### ABMIL + ECSA plug-in (Phase 3)
parser.add_argument('--use_ecsa', action='store_true', default=False,
                     help='ABMIL/host 모델에 ECSA plug-in 사용 (model_type=abmil 일 때만 유효)')
### AEM-style negative entropy auxiliary loss (Zhang et al. 2024)
parser.add_argument('--aem_lambda', type=float, default=0.0,
                     help='AEM (negative entropy) loss weight. 0.0 = off (default). '
                          'AEM paper uses 0.001~0.2 depending on dataset.')
### ACMIL config (model_type=acmil)
parser.add_argument('--acmil_n_token', type=int, default=5, help='ACMIL: # MBA branches')
parser.add_argument('--acmil_n_masked', type=int, default=10, help='ACMIL: STKIM top-k cardinality')
parser.add_argument('--acmil_mask_drop', type=float, default=0.6, help='ACMIL: STKIM drop probability')
parser.add_argument('--acmil_diff_w', type=float, default=1.0, help='ACMIL: diversity loss weight (n_token>1)')
parser.add_argument('--acmil_sub_w', type=float, default=1.0, help='ACMIL: sub-classifier loss weight')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B,
                    'dynamic_k': args.dynamic_k,
                    'k_min': args.k_min,
                    'k_max': args.k_max,
                    'gamma': args.gamma,
                    'dk_method': args.dk_method,
                    'dk_inverse': args.dk_inverse,
                    'adaptive_k_range': args.adaptive_k_range,
                    'k_min_pct': args.k_min_pct,
                    'k_max_pct': args.k_max_pct,
                    'uncertainty_weight': args.uncertainty_weight,
                    'learnable_temp': args.learnable_temp,
                    'adaptive_temp': args.adaptive_temp,
                    'sparse_topk': args.sparse_topk,
                    'attn_norm': args.attn_norm})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'task_camelyon16':
    args.n_classes=2
    # feature extractor에 따라 data_dir과 embed_dim 자동 설정
    fe_config = {
        'resnet50':   {'suffix': 'camelyon16_resnet_features',  'embed_dim': 1024},
        'ctranspath': {'suffix': 'camelyon16_ctrans_features',  'embed_dim': 768},
        'uni':        {'suffix': 'camelyon16_uni_features',     'embed_dim': 1024},
        'virchow':    {'suffix': 'camelyon16_virchow_features', 'embed_dim': 2560},
        'conch':      {'suffix': 'camelyon16_conch_features',   'embed_dim': 512},
    }
    fe = fe_config[args.feature_extractor]
    args.embed_dim = fe['embed_dim']
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon16.csv',
                            data_dir= os.path.join(args.data_root_dir, fe['suffix']),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_tcga_nsclc':
    args.n_classes=2
    fe_config = {
        'resnet50':   {'suffix': 'tcga_nsclc_resnet_features',  'embed_dim': 1024},
        'ctranspath': {'suffix': 'tcga_nsclc_ctrans_features',  'embed_dim': 768},
        'uni':        {'suffix': 'tcga_nsclc_uni_features',     'embed_dim': 1024},
        'virchow':    {'suffix': 'tcga_nsclc_virchow_features', 'embed_dim': 2560},
        'conch':      {'suffix': 'tcga_nsclc_conch_features',   'embed_dim': 512},
    }
    fe = fe_config[args.feature_extractor]
    args.embed_dim = fe['embed_dim']
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_nsclc.csv',
                            data_dir= os.path.join(args.data_root_dir, fe['suffix']),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_camelyon17':
    args.n_classes = 4
    fe_config = {
        'resnet50':   {'suffix': 'cam17_resnet_features',  'embed_dim': 1024},
        'ctranspath': {'suffix': 'cam17_ctrans_features',  'embed_dim': 768},
        'uni':        {'suffix': 'cam17_uni_features',     'embed_dim': 1024},
        'virchow':    {'suffix': 'cam17_virchow_features', 'embed_dim': 2560},
        'conch':      {'suffix': 'cam17_conch_features',   'embed_dim': 512},
    }
    fe = fe_config[args.feature_extractor]
    args.embed_dim = fe['embed_dim']
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon17.csv',
                            data_dir= os.path.join(args.data_root_dir, fe['suffix']),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'negative':0, 'itc':1, 'micro':2, 'macro':3},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_bracs':
    args.n_classes = 3   # 3-class group: BT (benign), AT (atypical), MT (malignant)
    fe_config = {
        'resnet50':   {'suffix': 'bracs_rn50',     'embed_dim': 1024},
        'ctranspath': {'suffix': 'bracs_ctrans',   'embed_dim': 768},
        'uni':        {'suffix': 'bracs_uni',      'embed_dim': 1024},
        'virchow':    {'suffix': 'bracs_virchow',  'embed_dim': 2560},
        'conch':      {'suffix': 'bracs_conch',    'embed_dim': 512},
    }
    fe = fe_config[args.feature_extractor]
    args.embed_dim = fe['embed_dim']
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/bracs.csv',
                            data_dir= os.path.join(args.data_root_dir, fe['suffix']),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'BT':0, 'AT':1, 'MT':2},
                            label_col = 'group',
                            patient_strat=False,
                            ignore=[])

else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


