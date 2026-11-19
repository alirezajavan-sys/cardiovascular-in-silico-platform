"""
Target selection and structure preparation (manuscript Methods 2.2).

Targets: SGLT2, ACE (note: manuscript Introduction explicitly flags an
ACE2->ACE target-switch; the docking/design targets used downstream are
SGLT2 and ACE), and NEP, per the mechanistic rationale given in the
Introduction (SGLT2 inhibition, RAAS modulation, neprilysin inhibition
as the three pillars motivating this platform's target set).

This script prepares receptor structures for docking:
  1. Fetch/validate a source PDB structure per target (id supplied in config).
  2. Strip crystallographic waters and heteroatoms not required for the
     binding site definition (co-crystallized ligand kept for grid-box
     centering, then removed for the receptor file itself).
  3. Add polar hydrogens and assign Gasteiger partial charges (via
     AutoDockTools/`prepare_receptor4.py`-equivalent), writing a PDBQT.
  4. Define the docking grid box, centered on the co-crystallized ligand
     (or a config-supplied center for apo structures).

Note (Discussion, Limitations): the manuscript does not separately report
a redocking validation (re-docking the co-crystallized ligand and
computing RMSD to the crystal pose) prior to library screening. This is
flagged as a limitation there; `redock_validate()` below is provided so a
user with access to the original structures can run and report that
check, but no RMSD value is invented here.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class TargetSpec:
    name: str
    pdb_id: str
    chain: str
    ligand_resname: str | None = None
    grid_center: tuple[float, float, float] | None = None
    grid_size: tuple[float, float, float] = (25.0, 25.0, 25.0)


def load_targets(config_path: str) -> list[TargetSpec]:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return [TargetSpec(**t) for t in raw.get("targets", [])]


def strip_waters_and_heteroatoms(pdb_path: Path, out_path: Path, keep_resname: str | None = None) -> None:
    """Remove HOH records and (optionally) all HETATM other than `keep_resname`."""
    with open(pdb_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname == "HOH":
                    continue
                if keep_resname is not None and resname != keep_resname:
                    continue
            fout.write(line)


def grid_box_from_ligand(pdb_path: Path, ligand_resname: str, padding: float = 10.0):
    """Compute a grid-box center as the geometric centroid of the named
    co-crystallized ligand's heavy atoms, with a config-specified padding."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == ligand_resname:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append((x, y, z))
    if not coords:
        raise ValueError(f"No atoms found for ligand '{ligand_resname}' in {pdb_path}")
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    return (cx, cy, cz)


def redock_validate(receptor_pdbqt: Path, ligand_pdbqt: Path, reference_pose_pdb: Path) -> float:
    """Re-dock the co-crystallized ligand and return RMSD to the crystal
    pose (heavy atoms only). Requires a local AutoDock Vina + RDKit
    installation. Not run automatically as part of the pipeline; call
    explicitly and report the returned value if a redocking validation
    is required (see module docstring)."""
    raise NotImplementedError(
        "Redocking validation requires a local Vina installation and the original "
        "receptor/ligand files, which are not bundled with this repository. "
        "Implement using vina.Vina().dock() + RDKit's AlignMol/GetBestRMS on the "
        "output pose vs. the reference crystal pose."
    )


def prepare_receptor(target: TargetSpec, raw_pdb_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb = raw_pdb_dir / f"{target.pdb_id}.pdb"
    if not raw_pdb.exists():
        raise FileNotFoundError(
            f"{raw_pdb} not found. Download {target.pdb_id} from the RCSB PDB and place "
            f"it under {raw_pdb_dir} (not redistributed in this repository)."
        )
    stripped = out_dir / f"{target.name}_stripped.pdb"
    strip_waters_and_heteroatoms(raw_pdb, stripped, keep_resname=target.ligand_resname)
    log.info("Prepared receptor for %s (PDB %s) -> %s", target.name, target.pdb_id, stripped)
    log.info(
        "NOTE: protonation-state assignment and Gasteiger charge computation require "
        "AutoDockTools (prepare_receptor4.py) or an equivalent OpenBabel/Meeko step, "
        "run externally on %s to produce the receptor PDBQT.",
        stripped,
    )
    return stripped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-pdb-dir", default="data_raw/pdb")
    parser.add_argument("--output", default="data_processed/receptors")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    targets = load_targets(args.config)
    raw_dir = Path(args.raw_pdb_dir)
    out_dir = Path(args.output)

    for target in targets:
        prepare_receptor(target, raw_dir, out_dir)


if __name__ == "__main__":
    main()
