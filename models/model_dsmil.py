"""DSMIL wrapper (Lu et al. 2021) — adapted from official repo
   https://github.com/binli123/dsmil-wsi
Returns (logits, Y_prob, Y_hat, A, results_dict) for our training loop.

Optional plug-in: replace bag-level dense aggregation `B = A^T @ V` with our
top-K selection (entropy-driven, floor=8 / inverse / featad)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy_k_dsmil(A_col, floor=8, k_max_cap=0, inverse=False):
    """Per-class entropy-driven k. A_col: 1D softmax attention over N for one class."""
    A = A_col + 1e-12
    A = A / A.sum()
    N = A.shape[0]
    H = -(A * torch.log(A)).sum().item()
    if inverse:
        H_max = max(torch.log(torch.tensor(float(N))).item(), 1e-8)
        H_norm = max(0.0, min(1.0, H / H_max))
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


class DSMIL_FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, feats):
        return feats, self.fc(feats)


class DSMIL_BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):
        super().__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU())
        else:
            self.v = nn.Identity()
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # feats:NxK, c:NxC
        device = feats.device
        V = self.v(feats)  # NxV
        Q = self.q(feats).view(feats.shape[0], -1)  # NxQ
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # CxK
        q_max = self.q(m_feats)  # CxQ
        A = torch.mm(Q, q_max.transpose(0, 1))  # NxC
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        B = torch.mm(A.transpose(0, 1), V)  # CxV
        B = B.view(1, B.shape[0], B.shape[1])
        C = self.fcc(B).view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    """Two-stream MIL: instance + bag, max(instance_max, bag) at inference.

    plugin_method: 'none' | 'floor8' | 'inverse' | 'featad'
                   Replaces bag-level aggregation `B = A^T @ V` with our
                   entropy-driven top-K selection (per-class).
    """

    def __init__(self, embed_dim=512, n_classes=2, dropout=0.0,
                 plugin_method='none', plugin_floor=8, plugin_kc=500, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.i_classifier = DSMIL_FCLayer(embed_dim, n_classes)
        self.b_classifier = DSMIL_BClassifier(embed_dim, n_classes, dropout_v=dropout)
        self.plugin_method = plugin_method
        self.plugin_floor = plugin_floor
        self.plugin_kc = plugin_kc

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        # h: [N, embed_dim]
        feats, classes = self.i_classifier(h)        # NxK, NxC
        prediction_bag, A, B = self.b_classifier(feats, classes)  # 1xC, NxC, 1xCxV

        # === Plugin: replace bag-level aggregation with our k-selection ===
        ks_used = []
        if self.plugin_method != 'none':
            inverse = (self.plugin_method == 'inverse')
            kc = self.plugin_kc if self.plugin_method in ('inverse', 'featad') else 0
            B_new = []
            V_feats = self.b_classifier.v(feats)  # NxV
            for c in range(self.n_classes):
                A_c = A[:, c]
                k = _entropy_k_dsmil(A_c, floor=self.plugin_floor, k_max_cap=kc, inverse=inverse)
                ks_used.append(k)
                top_vals, top_idx = torch.topk(A_c, k, dim=0)
                top_vals = top_vals / (top_vals.sum() + 1e-8)
                sel_V = V_feats.index_select(0, top_idx)
                B_new.append((top_vals.unsqueeze(0) @ sel_V).squeeze(0))
            B = torch.stack(B_new, dim=0).view(1, self.n_classes, -1)  # 1xCxV
            prediction_bag = self.b_classifier.fcc(B).view(1, -1)

        # max-pool instance prediction → 1xC
        max_prediction, _ = torch.max(classes, 0)
        max_prediction = max_prediction.view(1, -1)

        # 0.5 mean of bag and max predictions (DSMIL training combines both losses; at inference take avg)
        slide_logits = 0.5 * (prediction_bag + max_prediction)
        Y_hat = torch.argmax(slide_logits, dim=1, keepdim=True)
        Y_prob = F.softmax(slide_logits, dim=1)

        if attention_only:
            return A.t()  # CxN (consistent with our other models)

        k_pool_avg = int(sum(ks_used) / max(1, len(ks_used))) if ks_used else h.shape[0]
        results_dict = {
            'instance_logits': classes,
            'bag_logits': prediction_bag,
            'max_logits': max_prediction,
            'attention': A,
            'num_patches': h.shape[0],
            'k_pool': k_pool_avg,
        }
        if return_features:
            results_dict['features'] = B.squeeze(0)
        return slide_logits, Y_prob, Y_hat, A.t(), results_dict
