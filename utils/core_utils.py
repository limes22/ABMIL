import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, UncertaintyLossWeight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Optionally also tracks best val_AUC checkpoint side-by-side (track_auc=True).
    Stopping decision is still made by val_loss (main pipeline 호환); val_AUC 는
    별도 ckpt 로 저장만 → 평가 시 비교 가능.
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False, track_auc=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.track_auc = track_auc
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        # val_AUC tracking (parallel to val_loss)
        self.val_auc_max = -np.inf
        self.best_auc_epoch = -1
        self.best_loss_epoch = -1

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt', val_auc=None):
        score = -val_loss

        # val_loss-best (controls stopping)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.best_loss_epoch = epoch
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.best_loss_epoch = epoch
            self.counter = 0

        # val_AUC-best (parallel checkpoint, no stopping decision)
        if self.track_auc and val_auc is not None and val_auc > self.val_auc_max:
            self.val_auc_max = val_auc
            self.best_auc_epoch = epoch
            auc_ckpt = ckpt_name.replace('.pt', '_valauc.pt')
            torch.save(model.state_dict(), auc_ckpt)
            if self.verbose:
                print(f'  [val_AUC best: {val_auc:.4f} @ epoch {epoch} → {os.path.basename(auc_ckpt)}]')

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if getattr(args, 'adaptive_temp', False):
            model_dict.update({'adaptive_temp': True})
        elif getattr(args, 'learnable_temp', False):
            model_dict.update({'learnable_temp': True})

        if getattr(args, 'sparse_topk', False):
            model_dict.update({'sparse_topk': True})

        # Phase 1: AdaptiveSparsePooling (gated dense+sparse blend)
        if getattr(args, 'adaptive_sparse_pool', False):
            model_dict.update({'adaptive_sparse_pool': True})
        # Phase 2: feature_adaptive (meta-flag — all 5 adaptive modules)
        if getattr(args, 'feature_adaptive', False):
            model_dict.update({'feature_adaptive': True})
        # Phase 3: bag-size-aware k_max cap
        kc = getattr(args, 'k_max_cap', 0)
        if kc > 0:
            model_dict.update({'k_max_cap': kc})
        # Inverse mapping: low entropy (sharp) → large k
        if getattr(args, 'dk_inverse', False):
            model_dict.update({'dk_inverse': True})
        # entropy_only: k = round(exp(H)), bypasses k_min/k_max/k_max_cap
        if getattr(args, 'entropy_only', False):
            model_dict.update({'entropy_only': True})
        # entropy_k_floor: minimum k for entropy_only (default 8)
        if getattr(args, 'entropy_k_floor', 8) != 8:
            model_dict.update({'entropy_k_floor': args.entropy_k_floor})
        # loglinear k formula
        if getattr(args, 'loglinear', False):
            model_dict.update({
                'loglinear': True,
                'loglinear_k_min': getattr(args, 'loglinear_k_min', 8),
                'loglinear_k_cap': getattr(args, 'loglinear_k_cap', 500),
                'loglinear_cap_frac': getattr(args, 'loglinear_cap_frac', 0.0),
            })
        # Pass HP knobs (k_min_pct, k_max_pct, gamma, dk_method) to AdaptiveSparsePooling
        # These were previously dropped by the wrapper, forcing fixed defaults.
        if getattr(args, 'adaptive_sparse_pool', False) or getattr(args, 'feature_adaptive', False):
            model_dict.update({
                'k_min_pct': getattr(args, 'k_min_pct', 0.001),
                'k_max_pct': getattr(args, 'k_max_pct', 0.01),
                'gamma': getattr(args, 'gamma', 1.0),
                'entropy_method': getattr(args, 'dk_method', 'v2'),
            })
        # Phase 1 ablation knobs (ece-v3)
        model_dict.update({
            'inverse_threshold': getattr(args, 'inverse_threshold', 1.0),
            'hybrid_floor': getattr(args, 'hybrid_floor', False),
            'hybrid_floor_alpha': getattr(args, 'hybrid_floor_alpha', 0.0),
            'hybrid_floor_min': getattr(args, 'hybrid_floor_min', 8),
            'blend_w_bias': getattr(args, 'blend_w_bias', 0.0),
            'pure_cap': getattr(args, 'pure_cap', False),
            'pure_cap_frac': getattr(args, 'pure_cap_frac', 0.0),
            'pure_cap_min': getattr(args, 'pure_cap_min', 8),
            'learnable_alpha': getattr(args, 'learnable_alpha', False),
            'learnable_alpha_init': getattr(args, 'learnable_alpha_init', 0.03),
            'learnable_alpha_temp': getattr(args, 'learnable_alpha_temp', 20.0),
            'learnable_alpha_min': getattr(args, 'learnable_alpha_min', 8),
            'learnable_alpha_hybrid': getattr(args, 'learnable_alpha_hybrid', False),
            'lsap_temp': getattr(args, 'lsap_temp', False),
            'lsap_eps': getattr(args, 'lsap_eps', 0.01),
        })

        # attention normalization (default 'softmax', alternatives: 'sparsemax', 'entmax15')
        attn_norm = getattr(args, 'attn_norm', 'softmax')
        if attn_norm != 'softmax':
            model_dict.update({'attn_norm': attn_norm})

        # GELU activation (CTransPath 등 zero-centered features 호환)
        if getattr(args, 'use_gelu', False):
            model_dict.update({'use_gelu': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif args.model_type == 'acmil':
        # ACMIL (Zhang et al. ECCV 2024) wrapper with optional ECSA plug-in
        from models.model_acmil import ACMILWithECSA
        acmil_kwargs = dict(
            embed_dim=args.embed_dim,
            D_inner=512, D_attn=128,
            n_classes=args.n_classes,
            n_token=getattr(args, 'acmil_n_token', 5),
            n_masked_patch=getattr(args, 'acmil_n_masked', 10),
            mask_drop=getattr(args, 'acmil_mask_drop', 0.6),
            droprate=args.drop_out,
            use_ecsa=getattr(args, 'use_ecsa', False),
            adaptive_sparse_pool=getattr(args, 'adaptive_sparse_pool', False),
            feature_adaptive=getattr(args, 'feature_adaptive', False),
            k_max_cap=getattr(args, 'k_max_cap', 0),
            dk_inverse=getattr(args, 'dk_inverse', False),
            entropy_only=getattr(args, 'entropy_only', False),
            entropy_k_floor=getattr(args, 'entropy_k_floor', 8),
            loglinear=getattr(args, 'loglinear', False),
            loglinear_k_min=getattr(args, 'loglinear_k_min', 8),
            loglinear_k_cap=getattr(args, 'loglinear_k_cap', 500),
            loglinear_cap_frac=getattr(args, 'loglinear_cap_frac', 0.0),
            k_min_pct=getattr(args, 'k_min_pct', 0.001),
            k_max_pct=getattr(args, 'k_max_pct', 0.01),
            gamma=getattr(args, 'gamma', 1.0),
            entropy_method=getattr(args, 'dk_method', 'v2'),
            inverse_threshold=getattr(args, 'inverse_threshold', 1.0),
            hybrid_floor=getattr(args, 'hybrid_floor', False),
            hybrid_floor_alpha=getattr(args, 'hybrid_floor_alpha', 0.0),
            hybrid_floor_min=getattr(args, 'hybrid_floor_min', 8),
            blend_w_bias=getattr(args, 'blend_w_bias', 0.0),
            pure_cap=getattr(args, 'pure_cap', False),
            pure_cap_frac=getattr(args, 'pure_cap_frac', 0.0),
            pure_cap_min=getattr(args, 'pure_cap_min', 8),
            learnable_alpha=getattr(args, 'learnable_alpha', False),
            learnable_alpha_init=getattr(args, 'learnable_alpha_init', 0.03),
            learnable_alpha_temp=getattr(args, 'learnable_alpha_temp', 20.0),
            learnable_alpha_min=getattr(args, 'learnable_alpha_min', 8),
            learnable_alpha_hybrid=getattr(args, 'learnable_alpha_hybrid', False),
            attn_norm=getattr(args, 'attn_norm', 'softmax'),
            lsap_temp=getattr(args, 'lsap_temp', False),
            lsap_eps=getattr(args, 'lsap_eps', 0.01),
        )
        if acmil_kwargs['use_ecsa']:
            acmil_kwargs['ecsa_kwargs'] = dict(
                c_min=4, c_max=32,
                k_min_pct=getattr(args, 'k_min_pct', 0.001),
                k_max_pct=getattr(args, 'k_max_pct', 0.01),
                gamma=getattr(args, 'gamma', 1.0),
                entropy_method=getattr(args, 'dk_method', 'v1'),
                inverse=getattr(args, 'dk_inverse', False),
            )
        model = ACMILWithECSA(**acmil_kwargs)

    elif args.model_type in ('abmil', 'abmil_official'):
        # ABMIL: 우리 구현 (CLAM Attn_Net_Gated 기반) 또는 공식 (Ilse et al. 2018) 구현
        if args.model_type == 'abmil':
            from models.model_abmil import ABMIL
            abmil_kwargs = dict(
                embed_dim=args.embed_dim,
                n_classes=args.n_classes,
                dropout=args.drop_out,
                size_arg=args.model_size,
                use_ecsa=getattr(args, 'use_ecsa', False),
                adaptive_sparse_pool=getattr(args, 'adaptive_sparse_pool', False),
                feature_adaptive=getattr(args, 'feature_adaptive', False),
                k_max_cap=getattr(args, 'k_max_cap', 0),
            dk_inverse=getattr(args, 'dk_inverse', False),
            entropy_only=getattr(args, 'entropy_only', False),
            entropy_k_floor=getattr(args, 'entropy_k_floor', 8),
            loglinear=getattr(args, 'loglinear', False),
            loglinear_k_min=getattr(args, 'loglinear_k_min', 8),
            loglinear_k_cap=getattr(args, 'loglinear_k_cap', 500),
            loglinear_cap_frac=getattr(args, 'loglinear_cap_frac', 0.0),
            k_min_pct=getattr(args, 'k_min_pct', 0.001),
            k_max_pct=getattr(args, 'k_max_pct', 0.01),
            gamma=getattr(args, 'gamma', 1.0),
            entropy_method=getattr(args, 'dk_method', 'v2'),
            inverse_threshold=getattr(args, 'inverse_threshold', 1.0),
            hybrid_floor=getattr(args, 'hybrid_floor', False),
            hybrid_floor_alpha=getattr(args, 'hybrid_floor_alpha', 0.0),
            hybrid_floor_min=getattr(args, 'hybrid_floor_min', 8),
            blend_w_bias=getattr(args, 'blend_w_bias', 0.0),
            pure_cap=getattr(args, 'pure_cap', False),
            pure_cap_frac=getattr(args, 'pure_cap_frac', 0.0),
            pure_cap_min=getattr(args, 'pure_cap_min', 8),
            learnable_alpha=getattr(args, 'learnable_alpha', False),
            learnable_alpha_init=getattr(args, 'learnable_alpha_init', 0.03),
            learnable_alpha_temp=getattr(args, 'learnable_alpha_temp', 20.0),
            learnable_alpha_min=getattr(args, 'learnable_alpha_min', 8),
            learnable_alpha_hybrid=getattr(args, 'learnable_alpha_hybrid', False),
            attn_norm=getattr(args, 'attn_norm', 'softmax'),
            lsap_temp=getattr(args, 'lsap_temp', False),
            lsap_eps=getattr(args, 'lsap_eps', 0.01),
            lsap_no_tau=getattr(args, 'lsap_no_tau', False),
            lsap_alpha=getattr(args, 'lsap_alpha', 1.5),
            )
        else:  # abmil_official
            from models.model_abmil_official import GatedAttention as ABMIL
            abmil_kwargs = dict(
                embed_dim=args.embed_dim,
                M=512, L=128,
                n_classes=args.n_classes,
                dropout=args.drop_out,
                use_ecsa=getattr(args, 'use_ecsa', False),
            )
        if abmil_kwargs['use_ecsa']:
            abmil_kwargs['ecsa_kwargs'] = dict(
                c_min=4, c_max=32,
                k_min_pct=getattr(args, 'k_min_pct', 0.001),
                k_max_pct=getattr(args, 'k_max_pct', 0.01),
                gamma=getattr(args, 'gamma', 1.0),
                entropy_method=getattr(args, 'dk_method', 'v1'),
                inverse=getattr(args, 'dk_inverse', False),
            )
        model = ABMIL(**abmil_kwargs)

    elif args.model_type == 'dsmil':
        from models.model_dsmil import DSMIL
        model = DSMIL(embed_dim=args.embed_dim, n_classes=args.n_classes,
                      dropout=args.drop_out,
                      plugin_method=getattr(args, 'plugin_method', 'none'),
                      plugin_floor=getattr(args, 'entropy_k_floor', 8),
                      plugin_kc=getattr(args, 'plugin_kc', 500))

    elif args.model_type == 'transmil':
        from models.model_transmil import TransMIL
        model = TransMIL(embed_dim=args.embed_dim, n_classes=args.n_classes,
                         dropout=args.drop_out)

    elif args.model_type == 'dtfd':
        from models.model_dtfd import DTFD_MIL
        model = DTFD_MIL(embed_dim=args.embed_dim, n_classes=args.n_classes,
                         mDim=512, dropout=args.drop_out,
                         numGroup=4, total_instance=4, distill_type='AFS',
                         plugin_method=getattr(args, 'plugin_method', 'none'),
                         plugin_floor=getattr(args, 'entropy_k_floor', 8),
                         plugin_kc=getattr(args, 'plugin_kc', 500))

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    # ECC-DI override: set k_min/k_max floors if user requested
    eccdi_k_min_floor = getattr(args, 'k_min_floor', None)
    eccdi_k_max_floor = getattr(args, 'k_max_floor', None)
    if eccdi_k_min_floor is not None and hasattr(model, 'adaptive_pool'):
        model.adaptive_pool.k_min_floor = eccdi_k_min_floor
        print(f'[ECC-DI] Set adaptive_pool.k_min_floor = {eccdi_k_min_floor}')
    if eccdi_k_max_floor is not None and hasattr(model, 'adaptive_pool'):
        model.adaptive_pool.k_max_floor = eccdi_k_max_floor
        print(f'[ECC-DI] Set adaptive_pool.k_max_floor = {eccdi_k_max_floor}')

    _ = model.to(device)
    print('Done!')
    print_network(model)

    # Uncertainty Loss Weight 초기화 (Kendall et al., CVPR 2018)
    use_uw = getattr(args, 'uncertainty_weight', False)
    if use_uw and args.model_type in ['clam_sb', 'clam_mb']:
        loss_weight_module = UncertaintyLossWeight().to(device)
        print('\nUsing Uncertainty Loss Weighting (Kendall et al., 2018)')
    else:
        loss_weight_module = None

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args, loss_weight_module=loss_weight_module)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True,
                                       track_auc=getattr(args, 'track_valauc', False))
    else:
        early_stopping = None
    print('Done!')

    # dynamic k 설정
    use_dynamic_k = getattr(args, 'dynamic_k', False)
    dk_k_min = getattr(args, 'k_min', 4)
    dk_k_max = getattr(args, 'k_max', 16)
    dk_gamma = getattr(args, 'gamma', 1.0)
    dk_method = getattr(args, 'dk_method', 'v1')
    dk_inverse = getattr(args, 'dk_inverse', False)
    dk_adaptive_range = getattr(args, 'adaptive_k_range', False)
    dk_k_min_pct = getattr(args, 'k_min_pct', 0.001)
    dk_k_max_pct = getattr(args, 'k_max_pct', 0.01)

    # Per-epoch alpha trajectory + diagnostic log (for Learnable α / hybrid monitoring)
    alpha_log_path = os.path.join(args.results_dir, f's_{cur}_alpha_traj.csv')
    has_learnable_alpha = hasattr(model, 'adaptive_pool') and getattr(model.adaptive_pool, 'learnable_alpha', False)
    has_diag = has_learnable_alpha and hasattr(model.adaptive_pool, 'reset_diag_stats')
    if has_learnable_alpha:
        with open(alpha_log_path, 'w') as fp:
            if has_diag:
                fp.write('epoch,alpha,sharp_ratio,cap_binding_ratio,mean_H_norm,mean_k,mean_kN_ratio,n_slides\n')
            else:
                fp.write('epoch,alpha\n')

    # Adaptive-τ state machine.
    #
    # Three knobs (all from args):
    #   adaptive_tau_warmup_start    when (which epoch) to begin the FIRST collection
    #   adaptive_tau_warmup_epochs   how many epochs each collection window is
    #   adaptive_tau_update_interval if >0, re-collect+re-set τ every N epochs after the first finalization
    #
    # Behavior:
    #   The first window forces inverse OFF (force_no_inverse=True) — at that
    #   point any pre-existing inverse_threshold is meaningless. Subsequent
    #   periodic windows do NOT force inverse OFF — they sample the trained-
    #   regime H_norm distribution while inverse keeps using the previously-
    #   computed τ.
    use_adaptive_tau = (
        getattr(args, 'adaptive_tau', False)
        and has_learnable_alpha
        and hasattr(model.adaptive_pool, 'start_warmup')
    )
    adaptive_tau_warmup = max(1, int(getattr(args, 'adaptive_tau_warmup_epochs', 1)))
    adaptive_tau_start = max(0, int(getattr(args, 'adaptive_tau_warmup_start', 0)))
    adaptive_tau_update = max(0, int(getattr(args, 'adaptive_tau_update_interval', 0)))
    adaptive_tau_target = float(getattr(args, 'adaptive_tau_target', 0.3))
    # Persistent state across epochs:
    at_collecting = False
    at_collect_end = -1
    at_next_start = adaptive_tau_start if use_adaptive_tau else None
    at_first_done = False
    if use_adaptive_tau:
        adaptive_tau_log_path = os.path.join(args.results_dir, f's_{cur}_adaptive_tau.csv')
        with open(adaptive_tau_log_path, 'w') as fp:
            fp.write('event,epoch,warmup_epochs,target_quantile,tau_set,n_warmup_slides\n')
        print(f'[adaptive_tau] start={adaptive_tau_start} warmup_epochs={adaptive_tau_warmup} '
              f'update_interval={adaptive_tau_update} target_q={adaptive_tau_target}')

    for epoch in range(args.max_epochs):
        # ---- adaptive_tau state transitions at the start of each epoch ----
        if use_adaptive_tau:
            # Open a new collection window?
            if (not at_collecting) and at_next_start is not None and epoch == at_next_start:
                model.adaptive_pool.start_warmup(force_no_inverse=(not at_first_done))
                at_collecting = True
                at_collect_end = epoch + adaptive_tau_warmup
                tag = 'first_warmup' if not at_first_done else 'periodic_collect'
                print(f'[adaptive_tau] {tag} starting at epoch {epoch} '
                      f'(force_no_inverse={not at_first_done})')
                with open(adaptive_tau_log_path, 'a') as fp:
                    fp.write(f"{tag}_start,{epoch},{adaptive_tau_warmup},{adaptive_tau_target},,\n")
            # Close current collection window?
            if at_collecting and epoch == at_collect_end:
                n_collected = len(getattr(model.adaptive_pool, '_warmup_h_norms', []))
                tau_set = model.adaptive_pool.finalize_warmup(adaptive_tau_target)
                print(f'[adaptive_tau] finalized at epoch {epoch}: τ={tau_set} '
                      f'(from {n_collected} slides, target_q={adaptive_tau_target})')
                with open(adaptive_tau_log_path, 'a') as fp:
                    tag = 'first_warmup_done' if not at_first_done else 'periodic_done'
                    fp.write(f"{tag},{epoch},{adaptive_tau_warmup},{adaptive_tau_target},"
                             f"{('' if tau_set is None else f'{tau_set:.6f}')},{n_collected}\n")
                at_collecting = False
                at_first_done = True
                # Schedule next periodic collection?
                if adaptive_tau_update > 0:
                    at_next_start = epoch + adaptive_tau_update
                else:
                    at_next_start = None

        # Reset diagnostic accumulators at start of epoch (training pass)
        if has_diag:
            model.adaptive_pool.reset_diag_stats()

        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,
                            dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method,
                            loss_weight_module=loss_weight_module, dk_inverse=dk_inverse,
                            adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir,
                dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method,
                loss_weight_module=loss_weight_module, dk_inverse=dk_inverse,
                adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,
                       aem_lambda=getattr(args, 'aem_lambda', 0.0),
                       acmil_diff_w=getattr(args, 'acmil_diff_w', 1.0),
                       acmil_sub_w=getattr(args, 'acmil_sub_w', 1.0),
                       )
            stop = validate(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

        # Log per-epoch learned alpha + diag stats (after train+val)
        if has_learnable_alpha:
            cur_alpha = model.adaptive_pool.get_learned_alpha()
            with open(alpha_log_path, 'a') as fp:
                if has_diag:
                    diag = model.adaptive_pool.get_diag_stats()
                    if diag:
                        fp.write(f"{epoch},{cur_alpha:.6f},{diag['sharp_ratio']:.4f},{diag['cap_binding_ratio']:.4f},"
                                 f"{diag['mean_H_norm']:.4f},{diag['mean_k']:.2f},{diag['mean_kN_ratio']:.4f},{diag['count']}\n")
                    else:
                        fp.write(f'{epoch},{cur_alpha:.6f},,,,,,\n')
                else:
                    fp.write(f'{epoch},{cur_alpha:.6f}\n')
            if writer is not None:
                writer.add_scalar('learnable_alpha/sigmoid', cur_alpha, epoch)
                if has_diag:
                    diag = model.adaptive_pool.get_diag_stats()
                    if diag:
                        writer.add_scalar('lra_diag/sharp_ratio', diag['sharp_ratio'], epoch)
                        writer.add_scalar('lra_diag/cap_binding_ratio', diag['cap_binding_ratio'], epoch)
                        writer.add_scalar('lra_diag/mean_kN_ratio', diag['mean_kN_ratio'], epoch)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes,
                                       dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method, dk_inverse=dk_inverse,
                                       adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes,
                                                              dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method, dk_inverse=dk_inverse,
                                                              adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # ── Early-stopping ablation: val_AUC-best checkpoint 별도 평가 ──
    if getattr(args, 'track_valauc', False) and args.early_stopping:
        valauc_ckpt = os.path.join(args.results_dir, "s_{}_checkpoint_valauc.pt".format(cur))
        if os.path.exists(valauc_ckpt):
            model.load_state_dict(torch.load(valauc_ckpt))
            _, val_error_aucbest, val_auc_aucbest, _ = summary(model, val_loader, args.n_classes,
                                                                dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method, dk_inverse=dk_inverse,
                                                                adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)
            _, test_error_aucbest, test_auc_aucbest, _ = summary(model, test_loader, args.n_classes,
                                                                  dynamic_k=use_dynamic_k, k_min=dk_k_min, k_max=dk_k_max, gamma=dk_gamma, dk_method=dk_method, dk_inverse=dk_inverse,
                                                                  adaptive_k_range=dk_adaptive_range, k_min_pct=dk_k_min_pct, k_max_pct=dk_k_max_pct)
            print(f'[val_AUC-best ckpt @ ep {early_stopping.best_auc_epoch}]  val_AUC={val_auc_aucbest:.4f}, test_AUC={test_auc_aucbest:.4f}')
            # Inject into results_dict for downstream summary.csv
            results_dict['_es_ablation_val_auc_aucbest'] = float(val_auc_aucbest)
            results_dict['_es_ablation_test_auc_aucbest'] = float(test_auc_aucbest)
            results_dict['_es_ablation_test_err_aucbest'] = float(test_error_aucbest)
            results_dict['_es_ablation_best_auc_epoch'] = int(early_stopping.best_auc_epoch)
            results_dict['_es_ablation_best_loss_epoch'] = int(early_stopping.best_loss_epoch)
            # restore the val_loss-best ckpt (main result)
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,
                    dynamic_k=False, k_min=4, k_max=16, gamma=1.0, dk_method='v1',
                    loss_weight_module=None, dk_inverse=False,
                    adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
    model.train()
    if loss_weight_module is not None:
        loss_weight_module.train()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # dynamic k 통계 수집
    k_values = []
    entropy_values = []
    # Always track bag size (N) and pool size (k_pool) for visibility
    N_values = []
    k_pool_values = []
    sparse_ratios = []

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True,
                                                         dynamic_k=dynamic_k, k_min=k_min, k_max=k_max, gamma=gamma,
                                                         dk_method=dk_method, dk_inverse=dk_inverse,
                                                         adaptive_k_range=adaptive_k_range, k_min_pct=k_min_pct, k_max_pct=k_max_pct)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        # ── Loss weighting: uncertainty (learnable) vs fixed ──
        if loss_weight_module is not None:
            total_loss = loss_weight_module(loss, instance_loss)
        else:
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        # dynamic k 로깅
        if dynamic_k and 'dynamic_k' in instance_dict:
            k_values.append(instance_dict['dynamic_k'])
            entropy_values.append(instance_dict['entropy_norm'])

        train_loss += loss_value
        # Track per-batch N (bag_size) and k_pool (top-k actually pooled)
        N_val = instance_dict.get('num_patches', data.size(0))
        kp_val = instance_dict.get('k_pool', N_val)
        sr_val = instance_dict.get('sparse_ratio', None)
        N_values.append(N_val)
        k_pool_values.append(kp_val)
        if sr_val is not None:
            sparse_ratios.append(sr_val)
        if (batch_idx + 1) % 20 == 0:
            dk_info = ''
            if dynamic_k and 'dynamic_k' in instance_dict:
                dk_info = ', k_dyn: {}, entropy: {:.3f}'.format(instance_dict['dynamic_k'], instance_dict['entropy_norm'])
            pool_info = ', N: {}, k_pool: {}'.format(N_val, kp_val)
            if sr_val is not None:
                pool_info += ', sparse_ratio: {:.3f}'.format(sr_val)
            uw_info = ''
            if loss_weight_module is not None:
                uw = loss_weight_module.get_weights()
                uw_info = ', w_bag: {:.3f}, w_inst: {:.3f}'.format(uw['w_bag'], uw['w_inst'])
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) +
                'label: {}'.format(label.item()) + pool_info + dk_info + uw_info)

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))

    if dynamic_k and k_values:
        print('  dynamic_k stats: mean={:.1f}, min={}, max={}, mean_entropy={:.3f}'.format(
            np.mean(k_values), min(k_values), max(k_values), np.mean(entropy_values)))
    # Bag/pool size summary (always — works for baseline / sparse / adaptive)
    if N_values:
        n_arr = np.array(N_values); k_arr = np.array(k_pool_values)
        ratio = (k_arr / np.maximum(n_arr, 1)).mean()
        print('  pool stats: N mean={:.1f} (min={}, max={}); k_pool mean={:.1f} (min={}, max={}); k/N ratio mean={:.4f}'.format(
            n_arr.mean(), int(n_arr.min()), int(n_arr.max()),
            k_arr.mean(), int(k_arr.min()), int(k_arr.max()), ratio))
    if sparse_ratios:
        sr_arr = np.array(sparse_ratios)
        print('  adaptive_pool sparse_ratio: mean={:.3f} (min={:.3f}, max={:.3f})'.format(
            sr_arr.mean(), sr_arr.min(), sr_arr.max()))

    # Uncertainty weight 로깅
    if loss_weight_module is not None:
        uw = loss_weight_module.get_weights()
        print('  uncertainty_weight: w_bag={:.4f}, w_inst={:.4f}, log_var_bag={:.4f}, log_var_inst={:.4f}'.format(
            uw['w_bag'], uw['w_inst'], uw['log_var_bag'], uw['log_var_inst']))

    # Learnable temperature 로깅
    if hasattr(model, 'learnable_temp') and model.learnable_temp:
        print('  temperature: {:.4f}'.format(model.get_temperature()))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        if dynamic_k and k_values:
            writer.add_scalar('train/dynamic_k_mean', np.mean(k_values), epoch)
            writer.add_scalar('train/entropy_mean', np.mean(entropy_values), epoch)
        if loss_weight_module is not None:
            uw = loss_weight_module.get_weights()
            writer.add_scalar('train/uw_w_bag', uw['w_bag'], epoch)
            writer.add_scalar('train/uw_w_inst', uw['w_inst'], epoch)
            writer.add_scalar('train/uw_log_var_bag', uw['log_var_bag'], epoch)
            writer.add_scalar('train/uw_log_var_inst', uw['log_var_inst'], epoch)
        if hasattr(model, 'learnable_temp') and model.learnable_temp:
            writer.add_scalar('train/temperature', model.get_temperature(), epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,
               aem_lambda = 0.0, acmil_diff_w = 1.0, acmil_sub_w = 1.0,
               ):
    """
    Vanilla MIL training loop. Supports:
      - aem_lambda > 0 : AEM-style negative entropy auxiliary loss
        (Zhang et al. 2024). Adds  λ · Σ_n a_n log a_n  to bag loss.
      - ACMIL model: sub_logits + attn_softmax in results_dict.
    """
    import torch.nn.functional as F
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    aem_loss_sum = 0.

    print('\n')
    N_values, k_pool_values, sparse_ratios = [], [], []
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, A_raw, results_dict = model(data)

        # Track N / k_pool / sparse_ratio (for ABMIL/ACMIL+adapool visibility)
        if isinstance(results_dict, dict):
            N_values.append(results_dict.get('num_patches', data.size(0)))
            k_pool_values.append(results_dict.get('k_pool', data.size(0)))
            sr = results_dict.get('sparse_ratio')
            if sr is not None: sparse_ratios.append(sr)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # ACMIL auxiliary losses (sub + diff)
        sub_logits = results_dict.get('sub_logits') if isinstance(results_dict, dict) else None
        attn_soft = results_dict.get('attn_softmax') if isinstance(results_dict, dict) else None
        if sub_logits is not None and acmil_sub_w > 0:
            n_token = sub_logits.shape[0]
            sub_loss = loss_fn(sub_logits, label.repeat(n_token))
            loss = loss + acmil_sub_w * sub_loss
        if attn_soft is not None and attn_soft.shape[0] > 1 and acmil_diff_w > 0:
            n_token = attn_soft.shape[0]
            diff_loss = torch.tensor(0., device=loss.device)
            for i in range(n_token):
                for j in range(i + 1, n_token):
                    diff_loss = diff_loss + F.cosine_similarity(
                        attn_soft[i].unsqueeze(0), attn_soft[j].unsqueeze(0), dim=-1
                    ).mean()
            diff_loss = diff_loss / (n_token * (n_token - 1) / 2)
            loss = loss + acmil_diff_w * diff_loss

        # AEM negative-entropy auxiliary loss
        if aem_lambda > 0 and A_raw is not None:
            # A_raw: [K, N] (raw logits). Convert to softmax distribution.
            attn_softmax_aem = F.softmax(A_raw, dim=-1)
            attn_logsoftmax_aem = F.log_softmax(A_raw, dim=-1)
            # Σ p log p = - H(A);  loss += λ * (-H) → minimizing loss → maximizing H
            neg_ent = torch.sum(attn_softmax_aem * attn_logsoftmax_aem) / attn_softmax_aem.shape[0]
            loss = loss + aem_lambda * neg_ent
            aem_loss_sum += neg_ent.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            extra = ''
            if N_values:
                extra = ', N: {}, k_pool: {}'.format(N_values[-1], k_pool_values[-1])
                if sparse_ratios: extra += ', sparse_ratio: {:.3f}'.format(sparse_ratios[-1])
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}{}'.format(batch_idx, loss_value, label.item(), data.size(0), extra))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    if aem_lambda > 0:
        print(f'  AEM neg-entropy mean: {aem_loss_sum/len(loader):.4f}')

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    if N_values:
        n_arr = np.array(N_values); k_arr = np.array(k_pool_values)
        ratio = (k_arr / np.maximum(n_arr, 1)).mean()
        print('  pool stats: N mean={:.1f} (min={}, max={}); k_pool mean={:.1f} (min={}, max={}); k/N ratio mean={:.4f}'.format(
            n_arr.mean(), int(n_arr.min()), int(n_arr.max()),
            k_arr.mean(), int(k_arr.min()), int(k_arr.max()), ratio))
    if sparse_ratios:
        sr_arr = np.array(sparse_ratios)
        print('  adaptive_pool sparse_ratio: mean={:.3f} (min={:.3f}, max={:.3f})'.format(
            sr_arr.mean(), sr_arr.min(), sr_arr.max()))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                       val_auc=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None,
                  dynamic_k=False, k_min=4, k_max=16, gamma=1.0, dk_method='v1',
                  loss_weight_module=None, dk_inverse=False,
                  adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
    model.eval()
    if loss_weight_module is not None:
        loss_weight_module.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True,
                                                             dynamic_k=dynamic_k, k_min=k_min, k_max=k_max, gamma=gamma,
                                                             dk_method=dk_method, dk_inverse=dk_inverse,
                                                             adaptive_k_range=adaptive_k_range, k_min_pct=k_min_pct, k_max_pct=k_max_pct)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        if loss_weight_module is not None:
            uw = loss_weight_module.get_weights()
            writer.add_scalar('val/uw_w_bag', uw['w_bag'], epoch)
            writer.add_scalar('val/uw_w_inst', uw['w_inst'], epoch)

    # Uncertainty weight 로깅
    if loss_weight_module is not None:
        uw = loss_weight_module.get_weights()
        print('  uncertainty_weight: w_bag={:.4f}, w_inst={:.4f}'.format(uw['w_bag'], uw['w_inst']))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)


    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, dynamic_k=False, k_min=4, k_max=16, gamma=1.0, dk_method='v1', dk_inverse=False,
            adaptive_k_range=False, k_min_pct=0.001, k_max_pct=0.01):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data,
                                                 dynamic_k=dynamic_k, k_min=k_min, k_max=k_max, gamma=gamma, dk_method=dk_method, dk_inverse=dk_inverse,
                                                 adaptive_k_range=adaptive_k_range, k_min_pct=k_min_pct, k_max_pct=k_max_pct)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
