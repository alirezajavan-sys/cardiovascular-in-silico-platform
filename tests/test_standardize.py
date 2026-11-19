"""Unit tests for src.data.standardize (Methods 2.1)."""
from src.data.standardize import (
    StandardizationConfig,
    _inchikey14,
    _ro5_violations,
    mol_to_graph_features,
    pchembl,
    standardize_library,
)
from rdkit import Chem


def test_ro5_violations_passes_for_aspirin():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert _ro5_violations(mol) == 0


def test_inchikey14_deduplicates_identical_structures():
    mol_a = Chem.MolFromSmiles("CCO")
    mol_b = Chem.MolFromSmiles("OCC")  # same molecule, different SMILES
    assert _inchikey14(mol_a) == _inchikey14(mol_b)


def test_standardize_library_deduplicates_and_filters_by_mw():
    config = StandardizationConfig(mw_min=40.0, mw_max=100.0)
    records = [
        {"id": "1", "smiles": "CCO"},        # ethanol, MW ~46, passes
        {"id": "2", "smiles": "OCC"},        # duplicate of ethanol
        {"id": "3", "smiles": "c1ccccc1CCCCCCCCCCCCCCCCCC"},  # large, fails MW
        {"id": "4", "smiles": "not a smiles"},  # unparseable
    ]
    df = standardize_library(records, config)
    assert len(df) == 1
    assert df.iloc[0]["id"] == "1"


def test_mol_to_graph_features_edge_index_is_symmetric():
    mol = Chem.MolFromSmiles("CCO")
    feats = mol_to_graph_features(mol)
    assert len(feats["atom_features"]) == mol.GetNumAtoms()
    # every bond contributes both (i,j) and (j,i)
    assert len(feats["edge_index"]) == 2 * mol.GetNumBonds()


def test_pchembl_matches_known_value():
    # 1 uM = 1000 nM -> pChEMBL = 6.0
    assert abs(pchembl(1000.0) - 6.0) < 1e-9
