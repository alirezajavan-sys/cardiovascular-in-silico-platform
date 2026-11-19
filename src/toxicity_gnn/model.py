"""
Multi-task GNN cardiotoxicity model (manuscript Methods 2.4 / Fig. 3).

Architecture, exactly as described:
  - 4 message-passing graph-convolution layers, hidden dimension 256
  - Attention-weighted graph readout (attention pooling)
  - 8 parallel prediction heads: {hERG, Nav1.5, Cav1.2, IKs} x {blocker classifier, pIC50 regressor}
  - BatchNorm + Dropout(0.3) between layers

This module is Module B in the platform: it is used both prospectively, embedded
in the generative-design reward function (Module C), and -- once the step flagged
in the manuscript's Results 3.3 / Discussion is completed -- retrospectively, to
screen Module A's virtual-screening hits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation

CHANNELS = ["hERG", "Nav1.5", "Cav1.2", "IKs"]
N_ATOM_FEATURES = 6   # atomic_num, degree, formal_charge, hybridization, is_aromatic, total_h
HIDDEN_DIM = 256
N_MP_LAYERS = 4
DROPOUT = 0.3


class GraphEncoder(nn.Module):
    """Shared 4-layer message-passing encoder with attention-weighted readout."""

    def __init__(self, in_dim: int = N_ATOM_FEATURES, hidden_dim: int = HIDDEN_DIM,
                 n_layers: int = N_MP_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout

        gate_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.readout = AttentionalAggregation(gate_nn)

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        graph_embedding = self.readout(h, batch)
        return graph_embedding


class ChannelHead(nn.Module):
    """One CiPA channel's pair of heads: blocker classifier + pIC50 regressor."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embedding):
        logit = self.classifier(embedding).squeeze(-1)
        pic50 = self.regressor(embedding).squeeze(-1)
        return logit, pic50


class MultiTaskCardiotoxicityGNN(nn.Module):
    """Full 8-head multi-task model (Fig. 3)."""

    def __init__(self, in_dim: int = N_ATOM_FEATURES, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim)
        self.heads = nn.ModuleDict({ch: ChannelHead(hidden_dim) for ch in CHANNELS})

    def forward(self, x, edge_index, batch):
        embedding = self.encoder(x, edge_index, batch)
        out = {}
        for ch in CHANNELS:
            logit, pic50 = self.heads[ch](embedding)
            out[ch] = {"blocker_logit": logit, "pic50": pic50}
        return out

    def herg_blocker_probability(self, x, edge_index, batch) -> torch.Tensor:
        """Convenience used by the Module C reward function (Methods 2.5):
        returns P(hERG blocker) in [0, 1]."""
        out = self.forward(x, edge_index, batch)
        return torch.sigmoid(out["hERG"]["blocker_logit"])


class FocalLoss(nn.Module):
    """Focal loss (gamma=2.0), used in place of weighted BCE for the classification
    heads (Methods 2.4), to address the ~1:28 class imbalance in CardioSafe."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


def multitask_loss(outputs: dict, targets: dict, lambda_focal: float = 1.0, lambda_mse: float = 0.5,
                    focal_gamma: float = 2.0) -> torch.Tensor:
    """L = lambda1 * L_focal + lambda2 * L_MSE, summed across the 4 channels
    (Methods 2.4). `targets[ch]` should provide 'blocker_label' (0/1, NaN if
    unmeasured for that channel) and 'pic50' (NaN if unmeasured)."""
    focal = FocalLoss(gamma=focal_gamma)
    total = torch.tensor(0.0, device=next(iter(outputs.values()))["blocker_logit"].device)
    for ch in CHANNELS:
        logit = outputs[ch]["blocker_logit"]
        pic50_pred = outputs[ch]["pic50"]
        label = targets[ch]["blocker_label"]
        pic50_true = targets[ch]["pic50"]

        cls_mask = ~torch.isnan(label)
        reg_mask = ~torch.isnan(pic50_true)

        if cls_mask.any():
            total = total + lambda_focal * focal(logit[cls_mask], label[cls_mask])
        if reg_mask.any():
            total = total + lambda_mse * F.mse_loss(pic50_pred[reg_mask], pic50_true[reg_mask])
    return total
