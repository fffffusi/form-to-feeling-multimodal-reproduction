# -*- coding: utf-8 -*-
"""
Supplementary diagnostics for common-method bias and functional diagnostic signals.

This script reproduces the auxiliary checks discussed in the manuscript:
1. Harman single-factor test using the public product-level dataset.
2. Functional-polarity diagnostics when review-level functional extraction output
   is available from the full pipeline.

Default public-data input:
  data_anonymized/product_master.csv

Optional full-pipeline input:
  data/extraction/llm_function_extractions.csv

Outputs:
  diagnostics/cmb_harman_single_factor.csv
  diagnostics/mf_polarity_diagnostics.csv            (optional)
  diagnostics/mf_polarity_regressions.csv            (optional)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "diagnostics"

MASTER_CANDIDATES = [
    REPO_ROOT / "data_anonymized" / "product_master.csv",
    REPO_ROOT / "data" / "processed" / "product_master.csv",
]

FUNCTION_EXTRACTION_CANDIDATES = [
    REPO_ROOT / "data" / "extraction" / "llm_function_extractions.csv",
    REPO_ROOT / "data" / "processed" / "llm_function_extractions.csv",
]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_product_id(value) -> str:
    """Normalize product ids such as 1-1, 1—1, or file-name variants."""
    text = str(value).strip().replace("—", "-").replace("–", "-")
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return f"{int(nums[0])}-{int(nums[1])}"
    return text


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def zscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df[columns].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    return pd.DataFrame(
        StandardScaler().fit_transform(data),
        columns=columns,
        index=data.index,
    )


def run_harman(master: pd.DataFrame) -> pd.DataFrame:
    """Run Harman's single-factor test on ME/MF subdimensions and sentiment."""
    cmb_cols = [f"me{i}_z" for i in range(1, 6)]
    cmb_cols += [f"mf{i}_z" for i in range(1, 8)]
    cmb_cols += ["sentiment_z"]
    require_columns(master, cmb_cols, "product_master.csv")

    z = zscore_frame(master, cmb_cols)
    pca = PCA(n_components=1)
    pca.fit(z)

    return pd.DataFrame(
        [{
            "test": "Harman single-factor test",
            "variables": ", ".join(cmb_cols),
            "n_products": int(z.shape[0]),
            "first_unrotated_factor_variance_pct": pca.explained_variance_ratio_[0] * 100,
            "threshold_pct": 50.0,
            "interpretation": "Below 50%; common method variance is unlikely to be the sole driver.",
        }]
    )


def standardized_ols(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    target: str,
) -> dict:
    """Fit standardized OLS and report the target coefficient with HC1 robust SE."""
    require_columns(df, [outcome] + predictors, "diagnostic regression input")
    z = zscore_frame(df, [outcome] + predictors)

    y = z[outcome].to_numpy()
    x_raw = z[predictors].to_numpy()
    x = np.column_stack([np.ones(len(z)), x_raw])

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    n, k = x.shape

    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = x.T @ ((resid ** 2)[:, None] * x)
    hc1_scale = n / (n - k)
    cov_hc1 = hc1_scale * xtx_inv @ meat @ xtx_inv
    se_hc1 = np.sqrt(np.diag(cov_hc1))

    target_idx = predictors.index(target) + 1
    t_value = beta[target_idx] / se_hc1[target_idx]
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), df=n - k))

    ssr = resid @ resid
    sst = (y - y.mean()) @ (y - y.mean())
    r2 = 1 - ssr / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    return {
        "outcome": outcome,
        "target": target,
        "predictors": " + ".join(predictors),
        "beta_std": beta[target_idx],
        "se_hc1": se_hc1[target_idx],
        "t_hc1": t_value,
        "p_hc1": p_value,
        "adj_r2": adj_r2,
        "n": int(n),
    }


def run_functional_polarity(master: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize functional polarity and test negative functional signals."""
    require_columns(master, ["product_id", "price", "n_reviews_total"], "product_master.csv")
    require_columns(fx, ["product_id", "dimension", "polarity"], "llm_function_extractions.csv")

    master = master.copy()
    master["product_id"] = master["product_id"].apply(normalize_product_id)

    fx = fx.copy()
    fx["product_id"] = fx["product_id"].apply(normalize_product_id)
    fx["polarity_num"] = pd.to_numeric(fx["polarity"], errors="coerce")
    fx = fx.dropna(subset=["polarity_num"])

    mf_agg = (
        fx.groupby("product_id")
        .agg(
            mf_match_n=("polarity_num", "size"),
            mf_pos_n=("polarity_num", lambda s: int((s > 0).sum())),
            mf_neg_n=("polarity_num", lambda s: int((s < 0).sum())),
            mf_neu_n=("polarity_num", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    mf_agg["negative_ratio_mf"] = mf_agg["mf_neg_n"] / mf_agg["mf_match_n"]
    mf_agg["positive_ratio_mf"] = mf_agg["mf_pos_n"] / mf_agg["mf_match_n"]
    mf_agg["ln_mf_neg_count"] = np.log1p(mf_agg["mf_neg_n"])

    noise_names = {"噪音控制", "Noise Control", "MF2 Noise Control", "MF2"}
    mf2 = fx[fx["dimension"].astype(str).isin(noise_names)].copy()
    mf2_agg = (
        mf2.groupby("product_id")
        .agg(
            mf2_match_n=("polarity_num", "size"),
            mf2_pos_n=("polarity_num", lambda s: int((s > 0).sum())),
            mf2_neg_n=("polarity_num", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    mf2_agg["negative_ratio_mf2"] = mf2_agg["mf2_neg_n"] / mf2_agg["mf2_match_n"]
    mf2_agg["ln_mf2_neg_count"] = np.log1p(mf2_agg["mf2_neg_n"])

    df = master.merge(mf_agg, on="product_id", how="left").merge(mf2_agg, on="product_id", how="left")
    fill_cols = [
        "mf_match_n", "mf_pos_n", "mf_neg_n", "mf_neu_n",
        "negative_ratio_mf", "positive_ratio_mf", "ln_mf_neg_count",
        "mf2_match_n", "mf2_pos_n", "mf2_neg_n",
        "negative_ratio_mf2", "ln_mf2_neg_count",
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    df["ln_price"] = np.log(pd.to_numeric(df["price"], errors="coerce").clip(lower=1))
    df["ln_reviews"] = np.log(pd.to_numeric(df["n_reviews_total"], errors="coerce").clip(lower=1))

    summary = pd.DataFrame(
        [{
            "n_products": int(df.shape[0]),
            "products_with_mf_matches": int((df["mf_match_n"] > 0).sum()),
            "mf_matches_total": int(df["mf_match_n"].sum()),
            "mf_positive_total": int(df["mf_pos_n"].sum()),
            "mf_negative_total": int(df["mf_neg_n"].sum()),
            "mf_neutral_total": int(df["mf_neu_n"].sum()),
            "mf_negative_share_total": df["mf_neg_n"].sum() / df["mf_match_n"].sum()
            if df["mf_match_n"].sum() else np.nan,
            "mf_negative_ratio_product_mean": df["negative_ratio_mf"].mean(),
            "products_with_mf2_matches": int((df["mf2_match_n"] > 0).sum()),
            "mf2_matches_total": int(df["mf2_match_n"].sum()),
            "mf2_positive_total": int(df["mf2_pos_n"].sum()),
            "mf2_negative_total": int(df["mf2_neg_n"].sum()),
            "mf2_negative_share_total": df["mf2_neg_n"].sum() / df["mf2_match_n"].sum()
            if df["mf2_match_n"].sum() else np.nan,
        }]
    )

    outcomes = [col for col in ["y_50_50", "y_favrate_only", "y_sentiment_only"] if col in df.columns]
    pcs = [f"z_pc{i}" for i in range(1, 8) if f"z_pc{i}" in df.columns]

    rows = []
    targets = ["negative_ratio_mf", "ln_mf_neg_count", "negative_ratio_mf2", "ln_mf2_neg_count"]
    for outcome in outcomes:
        for target in targets:
            rows.append(standardized_ols(df, outcome, [target, "ln_price", "ln_reviews"], target))
            if pcs:
                rows.append(standardized_ols(df, outcome, [target, "ln_price", "ln_reviews"] + pcs, target))

    return summary, pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    master_path = first_existing(MASTER_CANDIDATES)
    if master_path is None:
        raise FileNotFoundError(
            "No product-level master data found. Expected one of:\n"
            + "\n".join(str(p) for p in MASTER_CANDIDATES)
        )

    master = pd.read_csv(master_path, encoding="utf-8-sig")
    if "product_id" in master.columns:
        master["product_id"] = master["product_id"].apply(normalize_product_id)

    cmb = run_harman(master)
    cmb_path = OUT_DIR / "cmb_harman_single_factor.csv"
    cmb.to_csv(cmb_path, index=False, encoding="utf-8-sig")

    print("\n[Harman single-factor test]")
    print(cmb.to_string(index=False))
    print(f"Saved: {cmb_path}")

    fx_path = first_existing(FUNCTION_EXTRACTION_CANDIDATES)
    if fx_path is None:
        print("\n[Functional-polarity diagnostics]")
        print("Skipped: review-level functional extraction output is not included in the default public dataset.")
        print("Run scripts_extraction/llm_extract_function.py on user-supplied reviews to generate:")
        print("  data/extraction/llm_function_extractions.csv")
        return

    fx = pd.read_csv(fx_path, encoding="utf-8-sig")
    polarity_summary, polarity_reg = run_functional_polarity(master, fx)

    polarity_path = OUT_DIR / "mf_polarity_diagnostics.csv"
    reg_path = OUT_DIR / "mf_polarity_regressions.csv"
    polarity_summary.to_csv(polarity_path, index=False, encoding="utf-8-sig")
    polarity_reg.to_csv(reg_path, index=False, encoding="utf-8-sig")

    print("\n[Functional-polarity diagnostics]")
    print(polarity_summary.to_string(index=False))
    print(f"Saved: {polarity_path}")

    print("\n[Functional-polarity regressions]")
    print(polarity_reg.to_string(index=False))
    print(f"Saved: {reg_path}")


if __name__ == "__main__":
    main()
