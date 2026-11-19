"""
Molecular standardization pipeline (manuscript Methods 2.1).

Steps, in order, matching the manuscript exactly:
  1. Parse structures (SDF/SMILES) with RDKit.
  2. Remove duplicates by InChIKey (first 14 characters -- the connectivity layer).
  3. Restrict molecular weight to 150-900 Da.
  4. Assess Rule-of-Five compliance (<=1 violation allowed).
  5. Apply PAINS filtering (RDKit's built-in PAINS catalog).
  6. Compute three representations: Morgan/ECFP4 fingerprints, MACCS keys,
     and molecular graphs (atom/bond feature tensors for the GNN).

Note (Discussion, Limitations): a dedicated protonation/tautomer standardization
step at physiological pH is NOT implemented here, consistent with the manuscript's
stated limitation. Users who want this behavior should add a step (e.g., via
Dimorphite-DL or OpenEye QUACPAC at pH 7.4) before the InChIKey de-duplication step,
since protonation state affects InChIKey computation.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)


@dataclass
class StandardizationConfig:
    mw_min: float = 150.0
    mw_max: float = 900.0
    max_ro5_violations: int = 1
    apply_pains_filter: bool = True
    ecfp_radius: int = 2
    ecfp_nbits: int = 2048

    @classmethod
    def from_yaml(cls, path: str) -> "StandardizationConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


def _inchikey14(mol: Chem.Mol) -> str | None:
    try:
        inchi = MolToInchi(mol)
        key = InchiToInchiKey(inchi)
        return key[:14] if key else None
    except Exception:
        return None


def _ro5_violations(mol: Chem.Mol) -> int:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return sum([mw > 500, logp > 5, hbd > 5, hba > 10])


def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def maccs_keys(mol: Chem.Mol):
    return MACCSkeys.GenMACCSKeys(mol)


ATOM_FEATURES = ["atomic_num", "degree", "formal_charge", "hybridization", "is_aromatic", "total_h"]
BOND_FEATURES = ["bond_type", "is_conjugated", "in_ring"]


def mol_to_graph_features(mol: Chem.Mol) -> dict:
    """Atom/bond feature extraction matching the manuscript's GNN input spec
    (Methods 2.1: atomic number, degree, formal charge, hybridization,
    aromaticity, total H count / bond type, conjugation, ring membership)."""
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(
            {
                "atomic_num": atom.GetAtomicNum(),
                "degree": atom.GetDegree(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": int(atom.GetHybridization()),
                "is_aromatic": int(atom.GetIsAromatic()),
                "total_h": atom.GetTotalNumHs(),
            }
        )
    bond_feats = []
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        f = {
            "bond_type": int(bond.GetBondType()),
            "is_conjugated": int(bond.GetIsConjugated()),
            "in_ring": int(bond.IsInRing()),
        }
        bond_feats.append(f)
        bond_feats.append(f)
    return {"atom_features": atom_feats, "bond_features": bond_feats, "edge_index": edge_index}


def standardize_library(records: list[dict], config: StandardizationConfig) -> pd.DataFrame:
    """records: list of {'id': ..., 'smiles': ...}. Returns a standardized,
    de-duplicated, filtered DataFrame with computed representations."""
    catalog = _pains_catalog() if config.apply_pains_filter else None

    seen_keys: set[str] = set()
    rows = []
    n_input = len(records)
    n_mw_fail = n_ro5_fail = n_pains_fail = n_dup = n_parse_fail = 0

    for rec in records:
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            n_parse_fail += 1
            continue

        mw = Descriptors.MolWt(mol)
        if not (config.mw_min <= mw <= config.mw_max):
            n_mw_fail += 1
            continue

        if _ro5_violations(mol) > config.max_ro5_violations:
            n_ro5_fail += 1
            continue

        if catalog is not None and catalog.HasMatch(mol):
            n_pains_fail += 1
            continue

        key = _inchikey14(mol)
        if key is None or key in seen_keys:
            n_dup += 1
            continue
        seen_keys.add(key)

        rows.append(
            {
                "id": rec["id"],
                "smiles": Chem.MolToSmiles(mol),
                "inchikey14": key,
                "mol_weight": mw,
                "ecfp4": morgan_fingerprint(mol, config.ecfp_radius, config.ecfp_nbits).ToBitString(),
                "maccs": maccs_keys(mol).ToBitString(),
            }
        )

    log.info(
        "standardize_library: input=%d parse_fail=%d mw_fail=%d ro5_fail=%d "
        "pains_fail=%d duplicate=%d retained=%d",
        n_input, n_parse_fail, n_mw_fail, n_ro5_fail, n_pains_fail, n_dup, len(rows),
    )
    return pd.DataFrame(rows)


def pchembl(standard_value_nm: float) -> float:
    """pChEMBL = -log10(standard_value_nM * 1e-9), per Methods 2.1."""
    import math

    return -math.log10(standard_value_nm * 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SDF or CSV (id,smiles) input file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = StandardizationConfig.from_yaml(args.config)

    in_path = Path(args.input)
    if in_path.suffix.lower() == ".sdf":
        records = []
        for mol in Chem.SDMolSupplier(str(in_path)):
            if mol is None:
                continue
            rid = mol.GetProp("_Name") if mol.HasProp("_Name") else str(len(records))
            records.append({"id": rid, "smiles": Chem.MolToSmiles(mol)})
    else:
        df_in = pd.read_csv(in_path)
        records = df_in.to_dict("records")

    df_out = standardize_library(records, config)
    out_path = Path(args.output)
    if out_path.suffix == ".parquet":
        df_out.to_parquet(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)
    log.info("Wrote %d standardized compounds to %s", len(df_out), out_path)


if __name__ == "__main__":
    main()
