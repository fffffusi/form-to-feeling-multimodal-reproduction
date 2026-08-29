"""Joint-significance FDR analysis for 14 component-specific pathways.

For each pathway, the composite null is H0: a*b = 0. The pathway-level
joint-significance p value is max(p_a, p_b). Benjamini-Hochberg (BH) and
Benjamini-Yekutieli (BY) adjustments are then applied across all 14 pathways.

Effect estimates, bootstrap standard errors, and BCa confidence intervals are
read from the 5,000-replication Stata analysis. The normal-approximation p value
stored in the Stata intermediate file is diagnostic only and is not used for
FDR adjustment.

Run ``stata/robustness_analysis.do`` before running this script.

References
----------
MacKinnon et al. (2002), Psychological Methods, 7(1), 83-104.
https://doi.org/10.1037/1082-989X.7.1.83
Yzerbyt et al. (2018), Journal of Personality and Social Psychology,
115(6), 929-943. https://doi.org/10.1037/pspa0000132
Benjamini and Hochberg (1995), JRSS B, 57(1), 289-300.
https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
Benjamini and Yekutieli (2001), Annals of Statistics, 29(4), 1165-1188.
https://doi.org/10.1214/aos/1013699998
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_anonymized" / "product_master.dta"
OUT = PROJECT_ROOT / "stata" / "output_robustness"
INDIRECT_RAW = OUT / "indirect_14paths_raw.csv"
BOOTSTRAP_CI_LOG = OUT / "indirect_14paths_bootstrap_ci.log"
VOLUME_LONG = OUT / "review_volume_sensitivity_long.csv"

RAW_PCS = [f"z_pc{i}" for i in range(1, 8)]
PCS = [f"z_pc{i}_s" for i in range(1, 8)]
RAW_CONTROLS = [
    "lnprice",
    "lnreviews",
    "has_discount",
    "free_shipping",
    "ln_imperfection",
    "ln_brand_avg_price",
    "recency_inv",
]
CONTROLS = [f"{name}_s" for name in RAW_CONTROLS]


def prepare_analysis_data(path: Path) -> pd.DataFrame:
    """Reproduce the transformations in the public Stata analysis."""
    data = pd.read_stata(path)
    required = {
        "product_id",
        "price",
        "n_reviews_total",
        "y_favrate_only",
        "m_e_avg",
        "m_f_avg",
        "has_discount",
        "free_shipping",
        "ln_imperfection",
        "ln_brand_avg_price",
        "recency_inv",
        *RAW_PCS,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Public data are missing required columns: {missing}")
    if data["product_id"].duplicated().any():
        raise ValueError("product_id must uniquely identify products")
    if data["n_reviews_total"].isna().any() or data["n_reviews_total"].le(0).any():
        raise ValueError("n_reviews_total must be positive and nonmissing")

    data = data.copy()
    data["lnprice"] = np.log(data["price"] + 1.0)
    data["lnreviews"] = np.log(data["n_reviews_total"] + 1.0)

    to_standardize = [
        "y_favrate_only",
        "m_e_avg",
        "m_f_avg",
        *RAW_PCS,
        *RAW_CONTROLS,
    ]
    for name in to_standardize:
        standard_deviation = data[name].std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            raise ValueError(f"Cannot standardize {name}: invalid standard deviation")
        data[f"{name}_s"] = (data[name] - data[name].mean()) / standard_deviation

    return data


def robust_ols(data: pd.DataFrame, outcome: str, predictors: list[str]):
    design = sm.add_constant(data[predictors], has_constant="add")
    return sm.OLS(data[outcome], design).fit(cov_type="HC1", use_t=True)


def fit_main_models(data: pd.DataFrame):
    affective = robust_ols(data, "m_e_avg_s", PCS + CONTROLS)
    functional = robust_ols(data, "m_f_avg_s", PCS + CONTROLS)
    outcome = robust_ols(
        data,
        "y_favrate_only_s",
        PCS + ["m_e_avg_s", "m_f_avg_s"] + CONTROLS,
    )
    return affective, functional, outcome


def parse_bootstrap_intervals(path: Path) -> pd.DataFrame:
    """Extract percentile and BCa intervals from Stata's bootstrap log."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
    rows: list[dict[str, float | str]] = []
    pending: dict[str, float | str] | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*(ind[EF]_pc\d)\s*\|(.+?)\(P\)\s*$", line)
        if match:
            values = [float(value) for value in re.findall(number, match.group(2))]
            if len(values) < 5:
                raise ValueError(f"Could not parse percentile interval line: {line}")
            pending = {
                "path_id": match.group(1),
                "percentile_ci_low": values[-2],
                "percentile_ci_high": values[-1],
            }
            continue

        if pending is not None and "(BCa)" in line:
            values = [
                float(value)
                for value in re.findall(number, line.split("(BCa)")[0])
            ]
            if len(values) < 2:
                raise ValueError(f"Could not parse BCa interval line: {line}")
            pending["bca_ci_low"] = values[-2]
            pending["bca_ci_high"] = values[-1]
            rows.append(pending)
            pending = None

    intervals = pd.DataFrame(rows)
    if len(intervals) != 14 or intervals["path_id"].duplicated().any():
        raise ValueError(
            f"Expected 14 unique bootstrap intervals, found {len(intervals)}"
        )
    return intervals


def load_indirect_results() -> pd.DataFrame:
    required_files = [INDIRECT_RAW, BOOTSTRAP_CI_LOG]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Run stata/robustness_analysis.do first. Missing: "
            + ", ".join(missing_files)
        )

    raw = pd.read_csv(INDIRECT_RAW)
    required_columns = {"path_id", "channel", "pc", "estimate", "boot_se"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing Stata output columns: {sorted(missing_columns)}")
    if len(raw) != 14 or raw["path_id"].duplicated().any():
        raise ValueError("Expected exactly 14 unique Stata indirect pathways")

    intervals = parse_bootstrap_intervals(BOOTSTRAP_CI_LOG)
    return raw.merge(intervals, on="path_id", how="left", validate="one_to_one")


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def make_joint_table(data: pd.DataFrame) -> pd.DataFrame:
    affective, functional, outcome = fit_main_models(data)
    source = load_indirect_results()

    rows = []
    for channel_code, channel_label, a_model, b_term in [
        ("E", "Affective evaluation", affective, "m_e_avg_s"),
        ("F", "Functional evaluation", functional, "m_f_avg_s"),
    ]:
        p_b = float(outcome.pvalues[b_term])
        b_coefficient = float(outcome.params[b_term])
        b_robust_se = float(outcome.bse[b_term])
        for pc in range(1, 8):
            path_id = f"ind{channel_code}_pc{pc}"
            term = f"z_pc{pc}_s"
            record = source.loc[source["path_id"].eq(path_id)].iloc[0]
            p_a = float(a_model.pvalues[term])
            rows.append(
                {
                    "path_id": path_id,
                    "channel": channel_label,
                    "pc": pc,
                    "path_label": (
                        f"PC{pc} -> {'ME' if channel_code == 'E' else 'MF'} "
                        "-> Y_platform"
                    ),
                    "estimate": float(record["estimate"]),
                    "boot_se": float(record["boot_se"]),
                    "bca_ci_low": float(record["bca_ci_low"]),
                    "bca_ci_high": float(record["bca_ci_high"]),
                    "a_coefficient": float(a_model.params[term]),
                    "a_robust_se": float(a_model.bse[term]),
                    "p_a": p_a,
                    "b_coefficient": b_coefficient,
                    "b_robust_se": b_robust_se,
                    "p_b": p_b,
                    "p_joint": max(p_a, p_b),
                }
            )

    table = pd.DataFrame(rows)
    table["p_bh_fdr"] = multipletests(table["p_joint"], method="fdr_bh")[1]
    table["p_by_fdr"] = multipletests(table["p_joint"], method="fdr_by")[1]
    table["bh_significant_0_05"] = table["p_bh_fdr"].le(0.05)
    table["result"] = np.where(
        table["bh_significant_0_05"],
        "Survives BH-FDR (adjusted p < 0.05)",
        "Does not survive BH-FDR",
    )
    return table


def four_core_paths(data: pd.DataFrame, label: str, order: int) -> list[dict]:
    affective, functional, outcome = fit_main_models(data)
    definitions = [
        ("PC3 -> ME", affective, "z_pc3_s"),
        ("PC6 -> MF", functional, "z_pc6_s"),
        ("ME -> Y_platform", outcome, "m_e_avg_s"),
        ("MF -> Y_platform", outcome, "m_f_avg_s"),
    ]
    rows = []
    for path_order, (path, model, term) in enumerate(definitions, start=1):
        p_value = float(model.pvalues[term])
        rows.append(
            {
                "spec_order": order,
                "specification": label,
                "path_order": path_order,
                "path": path,
                "coefficient": float(model.params[term]),
                "robust_se": float(model.bse[term]),
                "p_value": p_value,
                "stars": significance_stars(p_value),
                "N": int(model.nobs),
                "source": "data_anonymized/product_master.dta; OLS with HC1 robust SE",
            }
        )
    return rows


def make_volume_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(four_core_paths(data, "All products", 0))
    rows.extend(
        four_core_paths(
            data.loc[data["n_reviews_total"].ge(20)].copy(),
            "n_reviews_total >= 20",
            20,
        )
    )

    if not VOLUME_LONG.exists():
        raise FileNotFoundError(
            "Run stata/robustness_analysis.do first. Missing: "
            + str(VOLUME_LONG)
        )
    prior = pd.read_csv(VOLUME_LONG)
    path_map = {
        "PC3 -> affective": (1, "PC3 -> ME"),
        "PC6 -> functional": (2, "PC6 -> MF"),
        "Affective -> favorable rate": (3, "ME -> Y_platform"),
        "Functional -> favorable rate": (4, "MF -> Y_platform"),
    }
    specification_map = {
        "reviews_ge50": (50, "n_reviews_total >= 50"),
        "reviews_ge100": (100, "n_reviews_total >= 100"),
        "review_weighted": (999, "Review-volume weighted proxy"),
    }
    for _, record in prior.iterrows():
        if record["path"] not in path_map:
            continue
        path_order, path = path_map[record["path"]]
        specification_order, specification = specification_map[
            record["specification"]
        ]
        p_value = float(record["p_value"])
        rows.append(
            {
                "spec_order": specification_order,
                "specification": specification,
                "path_order": path_order,
                "path": path,
                "coefficient": float(record["b"]),
                "robust_se": float(record["robust_se"]),
                "p_value": p_value,
                "stars": significance_stars(p_value),
                "N": int(record["N"]),
                "source": "Stata robustness_analysis.do; vce(robust)",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["spec_order", "path_order"], kind="stable"
    )


def main() -> None:
    data = prepare_analysis_data(DATA)
    joint = make_joint_table(data)
    volume = make_volume_summary(data)

    OUT.mkdir(parents=True, exist_ok=True)
    joint_path = OUT / "Table_C4_joint_significance_FDR.csv"
    volume_path = OUT / "Table10_review_volume_summary.csv"
    joint.to_csv(joint_path, index=False, encoding="utf-8-sig")
    volume.to_csv(volume_path, index=False, encoding="utf-8-sig")

    focal = joint.loc[joint["path_id"].eq("indE_pc3")].iloc[0]
    print("PC3 affective joint-significance result")
    print(
        focal[
            [
                "estimate",
                "bca_ci_low",
                "bca_ci_high",
                "p_a",
                "p_b",
                "p_joint",
                "p_bh_fdr",
                "p_by_fdr",
            ]
        ].to_string()
    )
    print(f"\nGenerated: {joint_path.relative_to(PROJECT_ROOT)}")
    print(f"Generated: {volume_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
