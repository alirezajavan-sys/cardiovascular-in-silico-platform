"""
Inference script for the multi-task cardiotoxicity GNN (manuscript Methods
2.4 / Results 3.3). Loads a trained checkpoint and scores a list of SMILES
across all four CiPA channels {hERG, Nav1.5, Cav1.2, IKs}.

Used both standalone (to screen a candidate list, e.g. Module A's
repurposing hits or Module C's generated candidates) and internally by
`src.generative_design.generate` via `MultiTaskCardiotoxicityGNN.
herg_blocker_probability()`.
"""
from __future__ import annotations

import argparse
import csv
import logging

import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from ..data.standardize import mol_to_graph_features
from .model import CHANNELS, MultiTaskCardiotoxicityGNN

log = logging.getLogger(__name__)

_HYBRIDIZATION_LEVELS = 8  # RDKit HybridizationType enum size, used only for reference/documentation


def smiles_to_data(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = mol_to_graph_features(mol)
    x = torch.tensor(
        [[a[k] for k in ("atomic_num", "degree", "formal_charge", "hybridization", "is_aromatic", "total_h")]
         for a in feats["atom_features"]],
        dtype=torch.float,
    )
    if feats["edge_index"]:
        edge_index = torch.tensor(feats["edge_index"], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def load_model(checkpoint_path: str, device: str = "cpu") -> MultiTaskCardiotoxicityGNN:
    model = MultiTaskCardiotoxicityGNN()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model: MultiTaskCardiotoxicityGNN, smiles_list: list[str], device: str = "cpu") -> list[dict]:
    results = []
    valid_data, valid_smiles = [], []
    for smi in smiles_list:
        data = smiles_to_data(smi)
        if data is None:
            results.append({"smiles": smi, "error": "unparseable_smiles"})
            continue
        valid_data.append(data)
        valid_smiles.append(smi)

    if not valid_data:
        return results

    batch = Batch.from_data_list(valid_data).to(device)
    outputs = model(batch.x, batch.edge_index, batch.batch)

    for i, smi in enumerate(valid_smiles):
        row = {"smiles": smi}
        for ch in CHANNELS:
            prob = torch.sigmoid(outputs[ch]["blocker_logit"][i]).item()
            pic50 = outputs[ch]["pic50"][i].item()
            row[f"{ch}_blocker_probability"] = prob
            row[f"{ch}_pic50_pred"] = pic50
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Text file, one SMILES per line, or CSV with a 'smiles' column")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.input.endswith(".csv"):
        import pandas as pd
        smiles_list = pd.read_csv(args.input)["smiles"].tolist()
    else:
        with open(args.input) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    model = load_model(args.checkpoint, args.device)
    results = predict_batch(model, smiles_list, args.device)

    fieldnames = ["smiles"] + [f"{ch}_{suffix}" for ch in CHANNELS for suffix in ("blocker_probability", "pic50_pred")] + ["error"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    log.info("Wrote %d predictions to %s", len(results), args.output)


if __name__ == "__main__":
    main()
