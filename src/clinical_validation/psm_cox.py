"""
Retrospective clinical cross-reference via propensity-score matching (PSM)
and Cox proportional hazards regression on MIMIC-IV (manuscript Methods 2.8
/ Results 3.6, Tables 6-7).

Requires MIMIC-IV v2.2 access under a signed PhysioNet data use agreement
(see README Data access); no MIMIC-IV data is bundled with or redistributed
by this repository.

Note (Discussion, Limitations): as stated in the manuscript, this is an
observational association analysis and does not establish causal
treatment efficacy. The manuscript's Methods do not separately report the
exact ICD-9/ICD-10 and MIMIC-IV `itemid`/formulary mapping used to define
drug exposure and the heart-failure cohort, nor the a priori statistical
power calculation for the achieved cohort size; both are flagged there as
limitations. The mapping logic below (`build_cohort`) is a documented
reference implementation using MIMIC-IV's standard `prescriptions` and
`diagnoses_icd` tables and should be reviewed/adjusted against the
original cohort-definition SQL before being treated as a reproduction of
the manuscript's exact cohort.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)


@dataclass
class ClinicalConfig:
    caliper: float = 0.2  # standard deviations of the logit propensity score
    match_ratio: int = 1  # 1:1 matching
    covariates: tuple = ("age", "sex", "charlson_comorbidity_index", "baseline_egfr", "baseline_lvef")
    alpha: float = 0.05

    @classmethod
    def from_yaml(cls, path: str) -> "ClinicalConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def build_cohort(prescriptions: pd.DataFrame, diagnoses: pd.DataFrame, target_drug_formulary: list[str],
                  hf_icd_codes: list[str]) -> pd.DataFrame:
    """Reference cohort-definition logic (see module docstring caveat).
    `prescriptions`: MIMIC-IV `prescriptions` table (subject_id, drug, ...).
    `diagnoses`: MIMIC-IV `diagnoses_icd` table (subject_id, icd_code, ...).
    Returns one row per subject with an `exposed` (0/1) flag."""
    hf_subjects = set(diagnoses.loc[diagnoses["icd_code"].isin(hf_icd_codes), "subject_id"])
    exposed_subjects = set(
        prescriptions.loc[prescriptions["drug"].str.upper().isin([d.upper() for d in target_drug_formulary]), "subject_id"]
    )
    cohort = pd.DataFrame({"subject_id": sorted(hf_subjects)})
    cohort["exposed"] = cohort["subject_id"].isin(exposed_subjects).astype(int)
    return cohort


def estimate_propensity_scores(cohort: pd.DataFrame, config: ClinicalConfig) -> pd.Series:
    X = cohort[list(config.covariates)].to_numpy()
    y = cohort["exposed"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return pd.Series(model.predict_proba(X)[:, 1], index=cohort.index, name="propensity_score")


def match_cohort(cohort: pd.DataFrame, propensity_scores: pd.Series, config: ClinicalConfig) -> pd.DataFrame:
    """Greedy nearest-neighbor matching on the logit propensity score,
    within a caliper of `config.caliper` standard deviations, matching
    Methods 2.8's stated PSM approach."""
    cohort = cohort.copy()
    logit_ps = np.log(propensity_scores / (1 - propensity_scores)).replace([np.inf, -np.inf], np.nan)
    cohort["logit_ps"] = logit_ps
    cohort = cohort.dropna(subset=["logit_ps"])

    caliper_width = config.caliper * cohort["logit_ps"].std()
    treated = cohort[cohort["exposed"] == 1]
    control = cohort[cohort["exposed"] == 0]

    if treated.empty or control.empty:
        log.warning("Empty treated or control group after propensity estimation; cannot match.")
        return cohort.iloc[0:0]

    nn = NearestNeighbors(n_neighbors=config.match_ratio).fit(control[["logit_ps"]])
    distances, indices = nn.kneighbors(treated[["logit_ps"]])

    matched_rows = []
    for t_idx, (dists, idxs) in enumerate(zip(distances, indices)):
        for d, c_idx in zip(dists, idxs):
            if d <= caliper_width:
                matched_rows.append(treated.iloc[t_idx])
                matched_rows.append(control.iloc[c_idx])

    matched = pd.DataFrame(matched_rows).drop_duplicates()
    log.info("Matched %d treated to %d total (treated+control) within caliper", len(treated), len(matched))
    return matched


def psm_diagnostics(pre_match: pd.DataFrame, matched: pd.DataFrame, config: ClinicalConfig) -> pd.DataFrame:
    """Standardized mean differences pre- and post-match per covariate
    (Table 6), the standard PSM balance diagnostic."""
    rows = []
    for cov in config.covariates:
        def smd(df):
            treated = df.loc[df["exposed"] == 1, cov]
            control = df.loc[df["exposed"] == 0, cov]
            pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
            return (treated.mean() - control.mean()) / pooled_sd if pooled_sd > 0 else np.nan

        rows.append({"covariate": cov, "smd_pre_match": smd(pre_match), "smd_post_match": smd(matched)})
    return pd.DataFrame(rows)


def cox_regression(matched: pd.DataFrame, duration_col: str, event_col: str, config: ClinicalConfig) -> pd.DataFrame:
    """Cox proportional hazards regression on the matched cohort (Table 7)."""
    cph = CoxPHFitter()
    cols = ["exposed", duration_col, event_col] + list(config.covariates)
    cph.fit(matched[cols], duration_col=duration_col, event_col=event_col)
    summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prescriptions", help="MIMIC-IV prescriptions.csv (obtained under PhysioNet DUA)")
    parser.add_argument("--diagnoses", help="MIMIC-IV diagnoses_icd.csv")
    parser.add_argument("--outcomes", help="Per-subject outcome table with duration/event columns")
    parser.add_argument("--output-dir", default="data_processed/clinical")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = ClinicalConfig.from_yaml(args.config)

    if not (args.prescriptions and args.diagnoses and args.outcomes):
        log.warning(
            "This step requires MIMIC-IV v2.2 access under a signed PhysioNet data "
            "use agreement (see README Data access). No MIMIC-IV data is bundled "
            "with this repository; supply --prescriptions/--diagnoses/--outcomes "
            "once obtained."
        )
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    prescriptions = pd.read_csv(args.prescriptions)
    diagnoses = pd.read_csv(args.diagnoses)
    outcomes = pd.read_csv(args.outcomes)

    cohort = build_cohort(prescriptions, diagnoses, cfg["target_drug_formulary"], cfg["hf_icd_codes"])
    cohort = cohort.merge(outcomes, on="subject_id", how="inner")

    ps = estimate_propensity_scores(cohort, config)
    matched = match_cohort(cohort, ps, config)
    diagnostics = psm_diagnostics(cohort, matched, config)
    cox_summary = cox_regression(matched, cfg["duration_col"], cfg["event_col"], config)

    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_dir / "psm_diagnostics.csv", index=False)
    cox_summary.to_csv(out_dir / "cox_regression_results.csv", index=False)
    log.info("Wrote PSM diagnostics and Cox regression results to %s", out_dir)


if __name__ == "__main__":
    main()
