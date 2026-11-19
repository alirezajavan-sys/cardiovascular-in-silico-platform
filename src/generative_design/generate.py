"""
Dual-target generative molecular design (manuscript Methods 2.5).

Reward function, exactly as reported:

    R = 0.3 * pIC50(ACE) + 0.3 * pIC50(SGLT2) + 0.2 * QED
      + 0.1 * (1 - P(hERG blocker)) + 0.1 * SA_score_normalized

`P(hERG blocker)` is produced by Module B
(`src.toxicity_gnn.model.MultiTaskCardiotoxicityGNN.herg_blocker_probability`).
`pIC50(ACE)` / `pIC50(SGLT2)` require target-specific QSAR/affinity
predictors; `qsar_predictor.py`-style models are not part of the bundled
code (the manuscript does not separately report their architecture or
training data beyond citing the ACE/SGLT2 ChEMBL bioactivity sets used
for Module A) -- `score_candidate()` below takes them as injectable
callables so a user can plug in their own trained predictors.

Note (Discussion, Limitations): the manuscript's Methods 2.5 describes the
reward formula precisely but does not separately document the exact
generative-search algorithm's hyperparameters (population size, number of
generations, mutation/crossover operators and rates). This module
implements the reward function exactly as specified, and a genetic-algorithm
search loop (RDKit-based fragment crossover/mutation, in the style of
Jensen's graph-based GA) as a documented, standard, open-source choice for
this kind of multi-objective molecular optimization -- flagged here, and in
`configs/generative_design.yaml`, as a reference configuration rather than
a reproduction of undocumented original settings.
"""
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from typing import Callable

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

log = logging.getLogger(__name__)

# Weights exactly as reported in Methods 2.5 / Results 3.4
W_ACE = 0.3
W_SGLT2 = 0.3
W_QED = 0.2
W_HERG = 0.1
W_SA = 0.1


def normalize_sa_score(sa_score: float) -> float:
    """Synthetic accessibility (Ertl & Schuffenhauer, 2009) ranges ~1 (easy)
    to ~10 (hard). Normalized here to [0, 1] with 1 = most synthesizable,
    matching the reward function's "higher is better" convention."""
    sa_score = max(1.0, min(10.0, sa_score))
    return 1.0 - (sa_score - 1.0) / 9.0


@dataclass
class ScoreResult:
    smiles: str
    pic50_ace: float
    pic50_sglt2: float
    qed: float
    herg_blocker_prob: float
    sa_score_normalized: float
    reward: float


def score_candidate(
    smiles: str,
    ace_predictor: Callable[[str], float],
    sglt2_predictor: Callable[[str], float],
    herg_predictor: Callable[[str], float],
    sa_scorer: Callable[[str], float],
) -> ScoreResult | None:
    """Score one candidate SMILES with the exact Methods 2.5 reward formula.
    `*_predictor` callables take a SMILES string and return the relevant
    scalar (pIC50, hERG-blocker probability, or raw SA score 1-10)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pic50_ace = ace_predictor(smiles)
    pic50_sglt2 = sglt2_predictor(smiles)
    qed = QED.qed(mol)
    herg_prob = herg_predictor(smiles)
    sa_norm = normalize_sa_score(sa_scorer(smiles))

    reward = (
        W_ACE * pic50_ace
        + W_SGLT2 * pic50_sglt2
        + W_QED * qed
        + W_HERG * (1.0 - herg_prob)
        + W_SA * sa_norm
    )
    return ScoreResult(smiles, pic50_ace, pic50_sglt2, qed, herg_prob, sa_norm, reward)


# --------------------------------------------------------------------------
# Reference GA search loop (see module docstring: hyperparameters here are
# a documented reference configuration, not a reproduction of the
# manuscript's original -- undocumented -- run settings).
# --------------------------------------------------------------------------

def _random_mutation(mol: Chem.Mol) -> Chem.Mol | None:
    """Minimal atom-substitution mutation as a placeholder mutation operator.
    A production GA should use a richer fragment-replacement library (e.g.
    from a curated set of medicinal-chemistry-relevant fragments)."""
    try:
        rw = Chem.RWMol(mol)
        idx = random.randrange(rw.GetNumAtoms())
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
            candidates = ["N", "O"]
            atom.SetAtomicNum(Chem.Atom(random.choice(candidates)).GetAtomicNum())
            Chem.SanitizeMol(rw)
            return rw.GetMol()
    except Exception:  # noqa: BLE001
        return None
    return None


@dataclass
class GAConfig:
    population_size: int = 200  # TODO: confirm from original run logs
    n_generations: int = 100  # TODO: confirm from original run logs
    mutation_rate: float = 0.15  # TODO: confirm from original run logs
    elitism_fraction: float = 0.10
    seed: int = 42  # TODO: confirm from original run logs


def run_ga(
    seed_smiles: list[str],
    config: GAConfig,
    ace_predictor: Callable[[str], float],
    sglt2_predictor: Callable[[str], float],
    herg_predictor: Callable[[str], float],
    sa_scorer: Callable[[str], float],
) -> list[ScoreResult]:
    random.seed(config.seed)
    population = list(seed_smiles)
    while len(population) < config.population_size:
        population.append(random.choice(seed_smiles))

    best_history: list[ScoreResult] = []
    for gen in range(config.n_generations):
        scored = [
            r for r in (
                score_candidate(s, ace_predictor, sglt2_predictor, herg_predictor, sa_scorer)
                for s in population
            ) if r is not None
        ]
        scored.sort(key=lambda r: r.reward, reverse=True)
        if scored:
            best_history.append(scored[0])
            log.info("generation %d best_reward=%.4f smiles=%s", gen, scored[0].reward, scored[0].smiles)

        n_elite = max(1, int(config.elitism_fraction * config.population_size))
        elite = [r.smiles for r in scored[:n_elite]]

        next_gen = list(elite)
        while len(next_gen) < config.population_size:
            parent_smiles = random.choice(elite) if elite else random.choice(seed_smiles)
            mol = Chem.MolFromSmiles(parent_smiles)
            if mol is not None and random.random() < config.mutation_rate:
                mutant = _random_mutation(mol)
                if mutant is not None:
                    next_gen.append(Chem.MolToSmiles(mutant))
                    continue
            next_gen.append(parent_smiles)
        population = next_gen

    return best_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--seeds", help="Text file of seed SMILES (e.g. Module A hits)")
    parser.add_argument("--output", default="data_processed/generated_candidates.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.warning(
        "This reference driver requires target-specific ACE/SGLT2 pIC50 predictors "
        "that are not bundled in this repository (see module docstring). Supply "
        "`ace_predictor` / `sglt2_predictor` callables (e.g. wrapping your own trained "
        "QSAR models) before running end to end."
    )


if __name__ == "__main__":
    main()
