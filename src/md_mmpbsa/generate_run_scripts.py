"""
GROMACS run-script generation and MM-PBSA automation wrapper (manuscript
Methods 2.7 / Results 3.2, Table 1).

For each candidate (top screening/docking hits), this generates a
standard GROMACS MD pipeline: energy minimization -> NVT equilibration ->
NPT equilibration -> production MD, followed by an `gmx_MMPBSA` end-state
free-energy estimate on the resulting trajectory.

Force field: CHARMM36m (as reported in Methods 2.7). GROMACS and
gmx_MMPBSA are external dependencies this script only orchestrates; the
actual simulations require a local GROMACS installation and are not run
by this repository.

Note (Discussion, Limitations): the manuscript reports single-replicate
MM-PBSA free energies (Table 1 footnote) and does not separately document
several simulation parameters requested by reviewers (energy-minimization
force tolerance, PME real-space cutoff, thermostat/barostat time constants,
the exact solvent/ion model, and the MM-PBSA internal/external dielectric
constants used). The `.mdp` templates below use the CHARMM36-recommended
defaults from the GROMACS documentation, and each reviewer-requested value
that is not confirmable from the manuscript is marked
`; TODO: confirm from original run logs` inline so a user with access to
the original simulation logs can fill in the exact values used.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

EM_MDP = """\
; Energy minimization -- CHARMM36m defaults
integrator      = steep
emtol           = 1000.0   ; kJ/mol/nm -- TODO: confirm from original run logs
emstep          = 0.01
nsteps          = 50000
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2      ; nm -- TODO: confirm from original run logs (PME real-space cutoff)
rvdw            = 1.2
pbc             = xyz
"""

NVT_MDP = """\
; NVT equilibration -- CHARMM36m defaults
integrator      = md
nsteps          = 50000    ; 100 ps at dt=0.002
dt              = 0.002
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1 0.1  ; TODO: confirm from original run logs
ref_t           = 310 310  ; K -- physiological temperature
pcoupl          = no
gen_vel         = yes
gen_temp        = 310
gen_seed        = -1
constraints     = h-bonds
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
pbc             = xyz
"""

NPT_MDP = """\
; NPT equilibration -- CHARMM36m defaults
integrator      = md
nsteps          = 50000    ; 100 ps at dt=0.002
dt              = 0.002
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1 0.1
ref_t           = 310 310
pcoupl          = C-rescale
pcoupltype      = isotropic
tau_p           = 2.0      ; TODO: confirm from original run logs
ref_p           = 1.0
compressibility = 4.5e-5
constraints     = h-bonds
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
pbc             = xyz
"""

PRODUCTION_MDP = """\
; Production MD -- length per manuscript Methods 2.7 (see config: production_ns)
integrator      = md
nsteps          = {n_steps}
dt              = 0.002
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1 0.1
ref_t           = 310 310
pcoupl          = C-rescale
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = 1.0
compressibility = 4.5e-5
constraints     = h-bonds
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
pbc             = xyz
nstxout-compressed = 5000
"""

MMPBSA_IN = """\
&general
  startframe=1, endframe={n_frames}, interval=1,
  verbose=2,
  # Dielectric constants: TODO: confirm from original run logs
  # (using gmx_MMPBSA defaults: indi=1, exdi=80 for the standard PB model)
/
&pb
  indi=1, exdi=80, istrng=0.150,
/
"""


def generate_run_scripts(candidate_id: str, out_dir: Path, production_ns: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(production_ns * 1000 / 0.002)  # ns -> ps -> steps at dt=2fs
    n_frames = max(1, int(production_ns * 10))  # illustrative frame count for MM-PBSA extraction

    (out_dir / "em.mdp").write_text(EM_MDP)
    (out_dir / "nvt.mdp").write_text(NVT_MDP)
    (out_dir / "npt.mdp").write_text(NPT_MDP)
    (out_dir / "production.mdp").write_text(PRODUCTION_MDP.format(n_steps=n_steps))
    (out_dir / "mmpbsa.in").write_text(MMPBSA_IN.format(n_frames=n_frames))

    run_sh = f"""#!/bin/bash
# Auto-generated GROMACS + gmx_MMPBSA pipeline for candidate: {candidate_id}
# Requires: GROMACS >= 2023, gmx_MMPBSA, and pre-built topology files
# (complex.gro, complex.top, index.ndx) for this candidate -- not generated
# by this script; use CHARMM-GUI or pdb2gmx with the CHARMM36m force field.
set -euo pipefail

gmx grompp -f em.mdp -c complex.gro -p complex.top -o em.tpr -maxwarn 1
gmx mdrun -deffnm em

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p complex.top -n index.ndx -o nvt.tpr -maxwarn 1
gmx mdrun -deffnm nvt

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p complex.top -n index.ndx -o npt.tpr -maxwarn 1
gmx mdrun -deffnm npt

gmx grompp -f production.mdp -c npt.gro -t npt.cpt -p complex.top -n index.ndx -o production.tpr -maxwarn 1
gmx mdrun -deffnm production

gmx_MMPBSA -O -i mmpbsa.in -cs production.tpr -ct production.xtc \\
  -cp complex.top -o FINAL_RESULTS_MMPBSA_{candidate_id}.dat
"""
    run_path = out_dir / "run_pipeline.sh"
    run_path.write_text(run_sh)
    run_path.chmod(0o755)
    log.info("Wrote MD/MM-PBSA run scripts for %s -> %s", candidate_id, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="CSV with an 'id' column (e.g. screening_hits.csv)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="data_processed/md_runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    production_ns = cfg.get("production_ns", 100)  # TODO: confirm from original run logs if unset in config

    candidates_df = pd.read_csv(args.candidates)
    out_root = Path(args.output)
    for _, row in candidates_df.iterrows():
        generate_run_scripts(str(row["id"]), out_root / str(row["id"]), production_ns)

    log.info("Generated run scripts for %d candidates under %s", len(candidates_df), out_root)


if __name__ == "__main__":
    main()
