"""TransMIL wrapper (Shao et al. NeurIPS 2021) — adapted from official repo
   https://github.com/szc19990412/TransMIL

Returns (logits, Y_prob, Y_hat, A, results_dict) for our training loop.
NOTE: requires `nystrom_attention` package."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim, dim_head=dim // 8, heads=8,
            num_landmarks=dim // 2, pinv_iterations=6,
            residual=True, dropout=0.1)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, embed_dim=512, n_classes=2, dropout=0.0, **kwargs):
        super().__init__()
        D = 512
        self.pos_layer = PPEG(dim=D)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, D), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=D)
        self.layer2 = TransLayer(dim=D)
        self.norm = nn.LayerNorm(D)
        self._fc2 = nn.Linear(D, n_classes)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        # h: [N, embed_dim] → [B=1, N, embed_dim]
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = self._fc1(h)                               # [1, N, D]

        # pad to square
        N = h.shape[1]
        H_, W_ = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add = H_ * W_ - N
        h = torch.cat([h, h[:, :add, :]], dim=1)       # [1, H*W, D]

        # cls token + 2 trans layers + PPEG
        cls_t = self.cls_token.expand(1, -1, -1).to(h.device)
        h = torch.cat([cls_t, h], dim=1)               # [1, 1+H*W, D]
        h = self.layer1(h)
        h = self.pos_layer(h, H_, W_)
        h = self.layer2(h)

        h = self.norm(h)
        cls = h[:, 0]                                  # [1, D]
        slide_logits = self._fc2(cls)                  # [1, n_classes]

        Y_hat = torch.argmax(slide_logits, dim=1, keepdim=True)
        Y_prob = F.softmax(slide_logits, dim=1)

        # No explicit per-patch attention exposed in TransMIL — return uniform
        A_dummy = torch.ones(1, N, device=h.device) / N
        if attention_only:
            return A_dummy

        results_dict = {'num_patches': N, 'cls_features': cls.detach()}
        if return_features:
            results_dict['features'] = cls
        return slide_logits, Y_prob, Y_hat, A_dummy, results_dict
