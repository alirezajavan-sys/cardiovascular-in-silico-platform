"""Unit tests for the Module C reward function (Methods 2.5)."""
from src.generative_design.generate import normalize_sa_score, score_candidate


def test_normalize_sa_score_bounds():
    assert abs(normalize_sa_score(1.0) - 1.0) < 1e-9   # easiest -> 1
    assert abs(normalize_sa_score(10.0) - 0.0) < 1e-9  # hardest -> 0
    assert 0.0 <= normalize_sa_score(5.5) <= 1.0


def test_score_candidate_matches_manuscript_formula():
    smiles = "CCO"  # ethanol, arbitrary valid molecule for a formula-correctness check

    result = score_candidate(
        smiles,
        ace_predictor=lambda s: 7.0,
        sglt2_predictor=lambda s: 8.0,
        herg_predictor=lambda s: 0.2,
        sa_scorer=lambda s: 2.0,  # normalized -> (1 - (2-1)/9) = 0.8889
    )
    assert result is not None

    expected = 0.3 * 7.0 + 0.3 * 8.0 + 0.2 * result.qed + 0.1 * (1 - 0.2) + 0.1 * result.sa_score_normalized
    assert abs(result.reward - expected) < 1e-9


def test_score_candidate_returns_none_for_invalid_smiles():
    result = score_candidate(
        "not_a_valid_smiles!!",
        ace_predictor=lambda s: 0.0,
        sglt2_predictor=lambda s: 0.0,
        herg_predictor=lambda s: 0.0,
        sa_scorer=lambda s: 5.0,
    )
    assert result is None
