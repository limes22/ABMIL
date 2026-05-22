"""
ABMIL with the OFFICIAL gated attention from Ilse et al. (ICML 2018).

Source: https://github.com/AMLab-Amsterdam/AttentionDeepMIL (model.py, GatedAttention)

Differences from the original:
  - The MNIST-specific Conv2d feature extractor (feature_extractor_part1/part2)
    is REMOVED. We assume frozen pre-extracted features (e.g. from UNI, Virchow,
    CONCH foundation models) of arbitrary embedding dimension D are passed in.
  - The hidden dimensions M (bag-feature dim) and L (attention-hidden dim) are
    parameterized to match D and our network sizing.
  - The single-output sigmoid classifier is replaced by an n-class cross-entropy
    classifier so that the model is compatible with our k-fold CV pipeline.
  - Optional ECSA plug-in (§4.2 of our paper) replaces the standard
    soft aggregation `Z = A @ H` with entropy-conditional sparse aggregation.

The attention computation itself is byte-identical to the official code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ecsa import ECSA


class GatedAttention(nn.Module):
    """
    Verbatim from Ilse et al. 2018, with feature-extractor and classifier
    replaced for the WSI-pretrained-feature setting.
    """

    def __init__(
        self,
        embed_dim: int = 512,           # D, dimension of pre-extracted features
        M: int = 512,                   # bag-feature dimension (kept = embed_dim)
        L: int = 128,                   # attention hidden dimension
        n_classes: int = 2,
        attention_branches: int = 1,
        dropout: float = 0.0,
        use_ecsa: bool = False,
        ecsa_kwargs=None,
    ):
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        # Feature projection (replaces the MNIST Conv2d extractor)
        # WSI features arrive at dim `embed_dim`. Project to bag-feature dim M.
        if embed_dim == M:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Sequential(
                nn.Linear(embed_dim, M),
                nn.ReLU(),
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Gated attention — verbatim from Ilse et al. 2018
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix U
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        # Classifier — adapted to n-class CE (instead of single-output sigmoid)
        self.classifier = nn.Linear(self.M * self.ATTENTION_BRANCHES, n_classes)

        # ECSA plug-in (§4.2 of our paper)
        self.use_ecsa = use_ecsa
        self.ecsa = ECSA(**(ecsa_kwargs or {})) if use_ecsa else None

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        """
        Args:
            h: [N, embed_dim] pre-extracted instance features.

        Returns:
            (logits, Y_prob, Y_hat, A_raw, results_dict) — compatible with our
            train_loop / validate / summary in utils/core_utils.py.
        """
        # Project features to bag dim M
        H = self.feature_proj(h)               # [N, M]
        H = self.dropout(H)

        # Gated attention (verbatim from Ilse et al. 2018)
        A_V = self.attention_V(H)              # [N, L]
        A_U = self.attention_U(H)              # [N, L]
        A = self.attention_w(A_V * A_U)        # [N, ATTENTION_BRANCHES]
        A = torch.transpose(A, 1, 0)           # [ATTENTION_BRANCHES, N]
        if attention_only:
            return A
        A_raw = A
        A_softmax = F.softmax(A, dim=1)        # softmax over N

        # Aggregation: standard soft (default) or ECSA sparse (plug-in)
        if self.use_ecsa:
            z, ecsa_info = self.ecsa(A_softmax, H)
        else:
            z = torch.mm(A_softmax, H)         # [ATTENTION_BRANCHES, M]
            ecsa_info = {}

        # Classifier
        logits = self.classifier(z)            # [ATTENTION_BRANCHES, n_classes]
        Y_hat = torch.argmax(logits, dim=1, keepdim=True)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {'ecsa': ecsa_info}
        if return_features:
            results_dict['features'] = z
        return logits, Y_prob, Y_hat, A_raw, results_dict
