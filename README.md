# Cardiovascular In Silico Platform

Reference implementation of the four-module computational platform described in:

> Javan, A. *An Integrated In Silico Platform for Cardiovascular Drug Discovery: Multi-Target Repurposing, hERG Cardiotoxicity Prediction, Dual-Target Molecular Design, and Network-Based Biomarker Identification.* Scientific Reports (under revision).

This repository implements, end to end, the methods described in the manuscript's Methods section: molecular standardization, multi-target virtual screening, a multi-task graph neural network (GNN) cardiotoxicity filter, dual-target generative molecular design, multilayer network pharmacology, MD/MM-PBSA refinement automation, and retrospective clinical cross-referencing.

**Status.** This is a reference implementation matching the architecture, parameters, and workflow exactly as described in the manuscript's Methods. It is provided to satisfy the journal's code-availability requirement and to enable reproduction of the pipeline's *methodology*. Some scripts require access to restricted or licensed source data (ChEMBL, DrugBank, MIMIC-IV, GEO) that must be obtained independently by the user under the applicable data use agreements (see Data Access below) — no restricted data is redistributed in this repository.

## Pipeline overview

```
data/           -> target_selection/  -> screening/            -> toxicity_gnn/  -> generative_design/
(ChEMBL/DrugBank)  (SGLT2/ACE2/NEP)      (AutoDock Vina)           (shared filter)   (ACE/SGLT2 reward fn)
                                              |                         ^                    |
                                              v                         |                    v
                                         md_mmpbsa/  ------------------->                network_pharmacology/
                                         (GROMACS)                                       (GEO + multilayer PPI)
                                              |
                                              v
                                    clinical_validation/
                                    (MIMIC-IV, PSM, Cox)
```

## Repository structure

```
src/
  data/                    # RDKit standardization, fingerprints, molecular graphs (Methods 2.1)
  screening/                # Target/structure prep + AutoDock Vina automation (Methods 2.2-2.3)
  toxicity_gnn/              # Multi-task GNN model, training loop, inference (Methods 2.4)
  generative_design/         # Conditional generation + reward function (Methods 2.5)
  network_pharmacology/      # DEG analysis + multilayer network construction (Methods 2.6)
  md_mmpbsa/                 # GROMACS run generation + MM-PBSA automation (Methods 2.7)
  clinical_validation/       # MIMIC-IV cross-reference, PSM, Cox regression (Methods 2.8)
configs/                    # YAML configs holding every parameter reported in Methods
tests/                       # Unit tests for standardization, reward function, network metrics
environment.yml
requirements.txt
LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate cvd-in-silico
# or: pip install -r requirements.txt
```

Key dependencies: `rdkit`, `torch`, `torch-geometric`, `pandas`, `scipy`, `scikit-learn`, `networkx`, `python-louvain`, `lifelines` (Cox regression is also runnable via R `survival` — see `clinical_validation/README.md`), `gmx_MMPBSA` (external, via GROMACS).

## Data access

This repository does not redistribute any restricted dataset. To reproduce the pipeline, obtain:

- **ChEMBL 36** — public, CC BY-SA 4.0: https://www.ebi.ac.uk/chembl/
- **DrugBank 6** — academic license required: https://go.drugbank.com/
- **CardioSafe** — public, MIT license: https://github.com/AppliedScientific/CardioSafe-benchmark
- **UnihERG_DB** — see citation in manuscript reference 16
- **MIMIC-IV v2.2** — restricted; requires PhysioNet CITI training + data use agreement: https://physionet.org/
- **GEO** GSE28829, GSE41571 — public: https://www.ncbi.nlm.nih.gov/geo/

Place downloaded files under `data_raw/` (gitignored) following the paths referenced in `configs/data_sources.yaml`.

## Quickstart (per module)

```bash
# 1. Standardize a compound library
python -m src.data.standardize --input data_raw/chembl36.sdf --config configs/standardization.yaml --output data_processed/chembl36_standardized.parquet

# 2. Prepare targets and run virtual screening
python -m src.screening.prepare_targets --config configs/targets.yaml
python -m src.screening.run_docking --library data_processed/chembl36_standardized.parquet --config configs/docking.yaml

# 3. Train / run the multi-task cardiotoxicity GNN
python -m src.toxicity_gnn.train --config configs/gnn_training.yaml
python -m src.toxicity_gnn.predict --input <smiles_file> --checkpoint <path> --output predictions.csv

# 4. Generative design (ACE/SGLT2 dual-target)
python -m src.generative_design.generate --config configs/generative_design.yaml --gnn-checkpoint <path>

# 5. Network pharmacology
python -m src.network_pharmacology.differential_expression --config configs/geo_datasets.yaml
python -m src.network_pharmacology.build_network --config configs/network.yaml

# 6. MD/MM-PBSA automation (requires local GROMACS installation)
python -m src.md_mmpbsa.generate_run_scripts --candidates data_processed/screening_hits.csv --config configs/md_mmpbsa.yaml

# 7. Retrospective clinical cross-reference (requires MIMIC-IV access)
python -m src.clinical_validation.psm_cox --config configs/clinical_validation.yaml
```

## Reproducibility notes

All hyperparameters used in the manuscript are stored in `configs/` as the single source of truth (rather than hard-coded), so that `configs/gnn_training.yaml`, for example, is what the manuscript's Methods 2.4 section describes in prose. Where a manuscript reviewer requested a parameter not currently recorded in our source materials (e.g., MD energy-minimization tolerance, PME real-space cutoff, MM-PBSA dielectric constants — see the manuscript's Discussion, "Limitations"), the corresponding config field is marked `# TODO: confirm from original run logs` rather than filled with an invented value.

## License

MIT License (see `LICENSE`). Third-party data remain subject to their own licenses (see Data access above).

## Citation

If you use this code, please cite the manuscript (citation to be finalized upon publication) and this repository (Zenodo DOI to be minted on acceptance).
