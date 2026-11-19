"""
Multi-target virtual screening via AutoDock Vina (manuscript Methods 2.3).

Docks a standardized compound library (from `src.data.standardize`) against
each prepared receptor (from `src.screening.prepare_targets`) and ranks hits.

This wraps the `vina` Python bindings (AutoDock Vina >= 1.2). Vina itself
is an external dependency (not installable via pip alone on all platforms);
see https://vina.scripps.edu/. If the `vina` package is not installed,
`dock_library()` raises with an explanatory message rather than silently
skipping compounds.

Note (Discussion, Limitations): as stated in the manuscript, Vina docking
scores are used here as a fast, approximate ranking/triage signal for
virtual screening, not as quantitative binding-affinity predictions --
consistent with published critiques of empirical scoring-function
accuracy (see manuscript Discussion and its cited references). Ligands
surviving this triage are the input to Module B (toxicity_gnn) and,
downstream, MD/MM-PBSA refinement (md_mmpbsa), not final affinity calls.
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)


@dataclass
class DockingConfig:
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    seed: int = 42
    top_n_per_target: int = 200

    @classmethod
    def from_yaml(cls, path: str) -> "DockingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _require_vina():
    try:
        from vina import Vina  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The 'vina' package (AutoDock Vina Python bindings) is required for "
            "docking and is not bundled with this repository. Install per "
            "https://autodock-vina.readthedocs.io/en/latest/installation.html"
        ) from e


def dock_one(receptor_pdbqt: str, ligand_pdbqt: str, center: tuple, box_size: tuple,
             config: DockingConfig) -> float:
    """Dock a single ligand against a single receptor; returns the best
    (lowest) predicted binding-energy score in kcal/mol."""
    _require_vina()
    from vina import Vina

    v = Vina(sf_name="vina", seed=config.seed)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(box_size))
    v.dock(exhaustiveness=config.exhaustiveness, n_poses=config.num_modes)
    energies = v.energies(n_poses=config.num_modes)
    return float(energies[0][0])


def dock_library(library_df: pd.DataFrame, receptor_pdbqt: str, center: tuple,
                  box_size: tuple, ligand_pdbqt_dir: Path, config: DockingConfig) -> pd.DataFrame:
    """Dock every compound in `library_df` (must have an 'id' column and
    corresponding pre-converted PDBQT files under `ligand_pdbqt_dir`)
    against one receptor. Returns a DataFrame of {id, smiles, docking_score}
    sorted best-first."""
    rows = []
    for _, rec in library_df.iterrows():
        ligand_pdbqt = ligand_pdbqt_dir / f"{rec['id']}.pdbqt"
        if not ligand_pdbqt.exists():
            log.warning("Missing ligand PDBQT for %s, skipping (run a SMILES->PDBQT "
                        "conversion step, e.g. via Meeko, before docking).", rec["id"])
            continue
        try:
            score = dock_one(receptor_pdbqt, str(ligand_pdbqt), center, box_size, config)
        except Exception as exc:  # noqa: BLE001
            log.warning("Docking failed for %s: %s", rec["id"], exc)
            continue
        rows.append({"id": rec["id"], "smiles": rec["smiles"], "docking_score_kcal_mol": score})

    df = pd.DataFrame(rows).sort_values("docking_score_kcal_mol", ascending=True).reset_index(drop=True)
    return df.head(config.top_n_per_target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Parquet/CSV from src.data.standardize")
    parser.add_argument("--config", required=True)
    parser.add_argument("--receptor-dir", default="data_processed/receptors")
    parser.add_argument("--ligand-pdbqt-dir", default="data_processed/ligands_pdbqt")
    parser.add_argument("--output", default="data_processed/screening_hits.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = DockingConfig.from_yaml(args.config)

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    targets = raw.get("targets", [])
    if not targets:
        raise ValueError(f"{args.config} must define a 'targets' list (name, pdbqt, center, box_size).")

    lib_path = Path(args.library)
    library_df = pd.read_parquet(lib_path) if lib_path.suffix == ".parquet" else pd.read_csv(lib_path)

    all_hits = []
    for t in targets:
        receptor_pdbqt = str(Path(args.receptor_dir) / t["pdbqt"])
        hits = dock_library(
            library_df, receptor_pdbqt, tuple(t["center"]), tuple(t["box_size"]),
            Path(args.ligand_pdbqt_dir), config,
        )
        hits["target"] = t["name"]
        all_hits.append(hits)
        log.info("Docked %s: %d hits retained", t["name"], len(hits))

    out_df = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    log.info("Wrote %d total docking hits to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
