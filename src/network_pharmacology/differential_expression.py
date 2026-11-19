"""
Differential gene expression analysis from GEO datasets (manuscript Methods
2.6), used as input to the multilayer network construction in
`build_network.py`.

Manuscript-cited datasets: GSE28829, GSE41571 (see README "Data access").

The manuscript's Methods describe a limma-style moderated t-test workflow,
which is native to R/Bioconductor. This module implements the equivalent
computation in Python (Welch's t-test per gene + Benjamini-Hochberg FDR
correction via `statsmodels`), which is standard practice for a limma-
equivalent when the rest of the pipeline is Python-based, and is noted as
such rather than claimed to be a literal reimplementation of limma's
empirical Bayes moderation. Users who require limma exactly can instead run
the included `differential_expression.R` equivalent script referenced
below and feed its output into `build_network.py`.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)


@dataclass
class DEConfig:
    padj_threshold: float = 0.05
    log2fc_threshold: float = 1.0
    fdr_method: str = "fdr_bh"

    @classmethod
    def from_yaml(cls, path: str) -> "DEConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def differential_expression(
    expr_matrix: pd.DataFrame, case_samples: list[str], control_samples: list[str], config: DEConfig
) -> pd.DataFrame:
    """`expr_matrix`: genes (rows) x samples (columns), already normalized
    (e.g. RMA for microarray, or log2-CPM for RNA-seq -- normalization is
    dataset-specific and should be done upstream per the GEO series' own
    platform, per Methods 2.6). Returns a DataFrame of {gene, log2fc, pvalue,
    padj, significant}."""
    case = expr_matrix[case_samples].to_numpy()
    control = expr_matrix[control_samples].to_numpy()

    mean_case = case.mean(axis=1)
    mean_control = control.mean(axis=1)
    log2fc = mean_case - mean_control  # assumes expr_matrix is already log2-scale

    t_stat, p_value = stats.ttest_ind(case, control, axis=1, equal_var=False, nan_policy="omit")

    _, padj, _, _ = multipletests(np.nan_to_num(p_value, nan=1.0), method=config.fdr_method)

    result = pd.DataFrame(
        {
            "gene": expr_matrix.index,
            "log2fc": log2fc,
            "t_statistic": t_stat,
            "pvalue": p_value,
            "padj": padj,
        }
    )
    result["significant"] = (result["padj"] < config.padj_threshold) & (
        result["log2fc"].abs() > config.log2fc_threshold
    )
    return result.sort_values("padj")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML listing GEO series IDs and case/control sample groups")
    parser.add_argument("--expr-matrix", help="Pre-downloaded, normalized expression matrix (genes x samples, CSV)")
    parser.add_argument("--output", default="data_processed/deg_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = DEConfig.from_yaml(args.config)

    if not args.expr_matrix:
        log.warning(
            "No --expr-matrix supplied. Download and normalize GSE28829 / GSE41571 "
            "from GEO (see README Data access) before running this step; this "
            "repository does not redistribute the raw series matrices."
        )
        return

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    case_samples = raw["case_samples"]
    control_samples = raw["control_samples"]

    expr_matrix = pd.read_csv(args.expr_matrix, index_col=0)
    result = differential_expression(expr_matrix, case_samples, control_samples, config)
    result.to_csv(args.output, index=False)
    log.info("Wrote %d DEG results (%d significant) to %s", len(result), int(result["significant"].sum()), args.output)


if __name__ == "__main__":
    main()
