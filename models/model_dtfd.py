"""DTFD-MIL wrapper (Zhang et al. CVPR 2022) — adapted from official repo
   https://github.com/hrzhang1123/DTFD-MIL

DTFD: Double-Tier Feature Distillation MIL.
Splits a bag into K pseudo-bags, computes per-pseudo-bag prediction (Tier-1),
distills features (AFS = Aggregated Feature Selection), then aggregates
distilled features for slide prediction (Tier-2).

Returns (logits, Y_prob, Y_hat, A, results_dict) for our training loop.
NOTE: Combined Tier-1 + Tier-2 loss requires our trainer to also use the
returned tier1_loss term; we expose it via results_dict['tier1_loss']."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DimReduction(nn.Module):
    def __init__(self, in_dim, out_dim=512, n_res=0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.ModuleList()
        for _ in range(n_res):
            self.res.append(nn.Sequential(
                nn.Linear(out_dim, out_dim, bias=False), nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim, bias=False), nn.ReLU(inplace=True)))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.res:
            x = x + layer(x)
        return x


class _AttentionGated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attn_w = nn.Linear(D, K)

    def forward(self, x, isNorm=True):
        V = self.attn_V(x); U = self.attn_U(x)
        A = self.attn_w(V * U).transpose(1, 0)        # KxN
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class _AttentionWithClassifier(nn.Module):
    """Tier-2: aggregate pseudo-bag features + classify."""
    def __init__(self, L=512, D=128, n_classes=2, dropout=0.0):
        super().__init__()
        self.attn = _AttentionGated(L=L, D=D, K=1)
        self.cls = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(L, n_classes))

    def forward(self, x):  # x: K x L (one feat per pseudo-bag)
        A = self.attn(x)                       # 1xK
        afeat = torch.mm(A, x)                 # 1xL
        return self.cls(afeat)                 # 1 x n_classes


class _Classifier1fc(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


def _get_cam_1d(classifier, features):
    """Compute CAM weights for distillation (max/min selection)."""
    # features: N x L; weight: n_classes x L
    W = classifier.fc.weight                                 # n_classes x L
    cams = features @ W.t()                                  # N x n_classes
    return cams                                              # higher cam → more class-relevant


def _entropy_k(A_softmax, floor=8, k_max_cap=0, inverse=False):
    """Our entropy-driven k-selection. A_softmax: 1D softmax attention over N."""
    A = A_softmax + 1e-12
    A = A / A.sum()
    N = A.shape[0]
    H = -(A * torch.log(A)).sum().item()
    if inverse:
        # forward: H_norm in [0,1]; inverse: 1 - H_norm
        H_max = max(torch.log(torch.tensor(float(N))).item(), 1e-8)
        H_norm = max(0.0, min(1.0, H / H_max))
        # use linear: k_min + (k_max - k_min) * (1 - H_norm)
        k_min = max(4, int(N * 0.001))
        k_max = max(32, int(N * 0.01))
        if k_max_cap > 0:
            k_max = min(k_max, k_max_cap)
        k_max = max(k_max, k_min)
        k = int(k_min + (k_max - k_min) * (1.0 - H_norm))
    else:
        k = int(round(torch.exp(torch.tensor(H)).item()))
        k = max(k, floor)
        if k_max_cap > 0:
            k = min(k, k_max_cap)
    k = min(k, max(1, (N - 1) // 2))
    return max(k, 1)


def _topk_aggregate(A_softmax, feats, k):
    """Take top-k by A, normalize, weighted sum. A:[N], feats:[N, D] → [D]."""
    if k >= A_softmax.shape[0]:
        return torch.einsum('n,nd->d', A_softmax, feats)
    top_vals, top_idx = torch.topk(A_softmax, k, dim=0)
    top_vals = top_vals / (top_vals.sum() + 1e-8)
    sel_feats = feats.index_select(0, top_idx)
    return torch.einsum('n,nd->d', top_vals, sel_feats)


class DTFD_MIL(nn.Module):
    """One-call DTFD: pseudo-bag forming + Tier1 + AFS + Tier2.

    distill_type: 'AFS' (aggregated feature, default) | 'MaxS' | 'MaxMinS'
    numGroup: number of pseudo-bags per slide.

    plugin_method: 'none' | 'floor8' | 'inverse' | 'featad' (post-hoc on Tier-1)
    """

    def __init__(self, embed_dim=512, n_classes=2, mDim=512, dropout=0.0,
                 numGroup=4, total_instance=4, distill_type='AFS',
                 plugin_method='none', plugin_floor=8, plugin_kc=500,
                 **kwargs):
        super().__init__()
        self.dimReduction = _DimReduction(embed_dim, mDim, n_res=0)
        self.attention   = _AttentionGated(L=mDim, D=128, K=1)
        self.classifier  = _Classifier1fc(mDim, n_classes, dropout)
        self.attCls      = _AttentionWithClassifier(L=mDim, D=128, n_classes=n_classes, dropout=dropout)
        self.n_classes  = n_classes
        self.numGroup   = numGroup
        self.total_inst = total_instance
        self.distill    = distill_type
        self.plugin_method = plugin_method
        self.plugin_floor = plugin_floor
        self.plugin_kc = plugin_kc

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        # h: [N, embed_dim]
        N = h.shape[0]
        midFeat = self.dimReduction(h)                       # N x mDim
        AA = self.attention(midFeat, isNorm=False).squeeze(0)  # N

        # Pseudo-bag formation
        feat_idx = torch.randperm(N, device=h.device)
        chunks = torch.chunk(feat_idx, self.numGroup)

        slide_pseudo_feats = []           # K x mDim (distilled)
        sub_preds = []                    # tier-1 logits per pseudo-bag

        ks_used = []
        for tidx in chunks:
            if len(tidx) == 0: continue
            sub_feat = midFeat.index_select(0, tidx)         # ni x mDim
            sub_AA = AA.index_select(0, tidx)                # ni
            sub_AA = F.softmax(sub_AA, dim=0)
            # === Plugin: replace dense aggregation with our k-selection ===
            if self.plugin_method == 'floor8':
                k = _entropy_k(sub_AA, floor=self.plugin_floor, k_max_cap=0, inverse=False)
                tattFeat_tensor = _topk_aggregate(sub_AA, sub_feat, k).unsqueeze(0)
                ks_used.append(k)
            elif self.plugin_method == 'inverse':
                k = _entropy_k(sub_AA, floor=self.plugin_floor, k_max_cap=self.plugin_kc, inverse=True)
                tattFeat_tensor = _topk_aggregate(sub_AA, sub_feat, k).unsqueeze(0)
                ks_used.append(k)
            elif self.plugin_method == 'featad':
                k = _entropy_k(sub_AA, floor=self.plugin_floor, k_max_cap=self.plugin_kc, inverse=False)
                tattFeat_tensor = _topk_aggregate(sub_AA, sub_feat, k).unsqueeze(0)
                ks_used.append(k)
            else:
                tattFeat_tensor = torch.einsum('n,nd->d', sub_AA, sub_feat).unsqueeze(0)  # 1 x mDim
                ks_used.append(sub_feat.shape[0])
            tPredict = self.classifier(tattFeat_tensor)      # 1 x n_classes
            sub_preds.append(tPredict)

            # Distillation
            patch_pred_logits = _get_cam_1d(self.classifier, sub_feat)
            patch_pred_softmax = F.softmax(patch_pred_logits, dim=1)
            top_n = min(self.total_inst, sub_feat.shape[0])
            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:top_n]
            if self.distill == 'MaxS':
                slide_d_feat = sub_feat.index_select(0, topk_idx_max).mean(0, keepdim=True)
            elif self.distill == 'MaxMinS':
                topk_idx_min = sort_idx[-top_n:]
                slide_d_feat = torch.cat(
                    [sub_feat.index_select(0, topk_idx_max),
                     sub_feat.index_select(0, topk_idx_min)], dim=0).mean(0, keepdim=True)
            else:  # AFS — use weighted aggregate from attention
                slide_d_feat = tattFeat_tensor

            slide_pseudo_feats.append(slide_d_feat)

        slide_pseudo_feats = torch.cat(slide_pseudo_feats, dim=0)  # K x mDim
        sub_preds = torch.cat(sub_preds, dim=0)                    # K x n_classes

        slide_logits = self.attCls(slide_pseudo_feats)             # 1 x n_classes
        Y_hat = torch.argmax(slide_logits, dim=1, keepdim=True)
        Y_prob = F.softmax(slide_logits, dim=1)

        # Tier-1 loss to be added by trainer
        if label is not None:
            tier1_label = label.repeat(sub_preds.shape[0])
            tier1_loss = F.cross_entropy(sub_preds, tier1_label)
        else:
            tier1_loss = None

        if attention_only:
            return AA.unsqueeze(0)                                  # 1 x N

        # Average k across pseudo-bags (for log visibility)
        k_pool_avg = int(sum(ks_used) / max(1, len(ks_used))) if ks_used else N

        results_dict = {
            'sub_preds': sub_preds,
            'tier1_loss': tier1_loss,
            'attention': AA,
            'num_patches': N,
            'k_pool': k_pool_avg,
        }
        if return_features:
            results_dict['features'] = slide_pseudo_feats
        return slide_logits, Y_prob, Y_hat, AA.unsqueeze(0), results_dict
