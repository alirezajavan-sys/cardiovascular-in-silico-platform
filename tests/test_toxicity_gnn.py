"""Unit tests for src.toxicity_gnn.model (Methods 2.4 / Fig. 3)."""
import torch
from torch_geometric.data import Batch, Data

from src.toxicity_gnn.model import (
    CHANNELS,
    FocalLoss,
    MultiTaskCardiotoxicityGNN,
    N_ATOM_FEATURES,
    multitask_loss,
)


def _dummy_batch(n_graphs: int = 3, n_atoms: int = 5):
    data_list = []
    for _ in range(n_graphs):
        x = torch.randn(n_atoms, N_ATOM_FEATURES)
        # a simple ring of edges so every graph is connected
        edges = [(i, (i + 1) % n_atoms) for i in range(n_atoms)]
        edges += [(j, i) for i, j in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list)


def test_forward_produces_all_eight_heads():
    model = MultiTaskCardiotoxicityGNN()
    batch = _dummy_batch()
    out = model(batch.x, batch.edge_index, batch.batch)
    assert set(out.keys()) == set(CHANNELS)
    for ch in CHANNELS:
        assert out[ch]["blocker_logit"].shape == (3,)
        assert out[ch]["pic50"].shape == (3,)


def test_herg_blocker_probability_in_unit_interval():
    model = MultiTaskCardiotoxicityGNN()
    batch = _dummy_batch()
    probs = model.herg_blocker_probability(batch.x, batch.edge_index, batch.batch)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_focal_loss_nonnegative():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0


def test_multitask_loss_handles_missing_labels_via_nan_mask():
    model = MultiTaskCardiotoxicityGNN()
    batch = _dummy_batch(n_graphs=4)
    out = model(batch.x, batch.edge_index, batch.batch)

    targets = {}
    for ch in CHANNELS:
        label = torch.tensor([0.0, 1.0, float("nan"), 0.0])
        pic50 = torch.tensor([5.0, float("nan"), 6.5, 7.0])
        targets[ch] = {"blocker_label": label, "pic50": pic50}

    loss = multitask_loss(out, targets)
    assert torch.isfinite(loss)
