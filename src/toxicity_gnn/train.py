"""
Training loop for the multi-task cardiotoxicity GNN (manuscript Methods 2.4).

Trains `MultiTaskCardiotoxicityGNN` (see `model.py`) on a labeled dataset
of molecular graphs with per-channel {hERG, Nav1.5, Cav1.2, IKs} blocker
labels and pIC50 values, following the CardioSafe benchmark's data split
(see manuscript Data Availability / README "Data access" for the source).

Loss: `multitask_loss` (focal loss for classification, gamma=2.0, summed
with MSE regression loss, lambda1=1.0 / lambda2=0.5), matching Methods 2.4.

Note (Discussion, Limitations): the manuscript does not separately report
the train/validation/test split random seed, the learning-rate schedule,
or the exact early-stopping patience used for the reported run. Config
defaults below are reasonable choices for this architecture and dataset
size and are marked in `configs/gnn_training.yaml` for the user to
confirm/replace against their original run logs.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from .model import CHANNELS, MultiTaskCardiotoxicityGNN, multitask_loss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_patience: int = 10  # TODO: confirm from original run logs
    early_stop_patience: int = 20  # TODO: confirm from original run logs
    lambda_focal: float = 1.0
    lambda_mse: float = 0.5
    focal_gamma: float = 2.0
    val_split: float = 0.10
    test_split: float = 0.10
    seed: int = 42  # TODO: confirm from original run logs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def split_dataset(dataset: list, config: TrainConfig):
    g = torch.Generator().manual_seed(config.seed)
    n = len(dataset)
    n_test = int(n * config.test_split)
    n_val = int(n * config.val_split)
    n_train = n - n_val - n_test
    return torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=g)


def run_epoch(model, loader, config: TrainConfig, optimizer=None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(config.device)
        outputs = model(batch.x, batch.edge_index, batch.batch)
        targets = {
            ch: {
                "blocker_label": getattr(batch, f"{ch}_label"),
                "pic50": getattr(batch, f"{ch}_pic50"),
            }
            for ch in CHANNELS
        }
        loss = multitask_loss(outputs, targets, config.lambda_focal, config.lambda_mse, config.focal_gamma)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(dataset: list, config: TrainConfig) -> MultiTaskCardiotoxicityGNN:
    """`dataset` is a list of `torch_geometric.data.Data` objects, each with
    `x`, `edge_index`, and, per channel, `{channel}_label` / `{channel}_pic50`
    attributes (NaN where unmeasured for that compound/channel)."""
    torch.manual_seed(config.seed)
    train_set, val_set, test_set = split_dataset(dataset, config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    model = MultiTaskCardiotoxicityGNN().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=config.lr_patience)

    best_val = float("inf")
    epochs_no_improve = 0
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(config.epochs):
        train_loss = run_epoch(model, train_loader, config, optimizer)
        val_loss = run_epoch(model, val_loader, config, optimizer=None)
        scheduler.step(val_loss)
        log.info("epoch %d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stop_patience:
                log.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val)
                break

    model.load_state_dict(torch.load(best_path, map_location=config.device))
    test_loss = run_epoch(model, test_loader, config, optimizer=None)
    log.info("Final held-out test loss: %.4f", test_loss)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", help="Path to a pre-built .pt list of torch_geometric Data objects "
                                           "(see src.data.standardize.mol_to_graph_features for feature spec)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = TrainConfig.from_yaml(args.config)

    if not args.dataset:
        raise SystemExit(
            "No --dataset given. Build a labeled graph dataset from CardioSafe "
            "(see README Data access) using src.data.standardize.mol_to_graph_features, "
            "then pass its path here."
        )
    dataset = torch.load(args.dataset)
    train(dataset, config)


if __name__ == "__main__":
    main()
