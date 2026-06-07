"""
compute_kansei_function_scores.py
======================================
Aggregate comment-level matches into product-level ME (5 dimensions) and MF (7 dimensions).

Affective dimensions (matches Appendix A.3 of the paper):
  ME1 Perceived Premiumness   (premium vs. cheap)
  ME2 Visual Solidity         (solid/substantial vs. flimsy)
  ME3 Minimalist Simplicity   (minimal/clean vs. cluttered)
  ME4 Aesthetic Refinement    (refined/beautiful vs. rough/ugly)
  ME5 Tech-Modernity          (tech-modern vs. old-fashioned)

Functional dimensions (matches Appendix A.4 of the paper):
  MF1 Lifting Performance
  MF2 Noise Control
  MF3 Load-Bearing Stability
  MF4 Material and Build Quality
  MF5 Odor and Environmental Safety
  MF6 Control and Interface Integration
  MF7 Workspace Size Adequacy

Inputs:
  data/extraction/llm_kansei_extractions.parquet    (long format)
  data/extraction/llm_function_extractions.parquet  (long format)
  product_reviews_clean.xlsx                        (N_total_reviews per product)
  lexicons/degree_words.json                        (degree_weight lookup table)
  data/satisfaction/product_satisfaction.csv        (optional: for merging Y variables)

Outputs (with z-scores + aggregate dimensions):
  data/processed/product_kansei_scores.csv          (200 x 19)
      product_id, n_reviews,
      ME1..ME5 (raw), ME1_z..ME5_z (standardized),
      cov_ME1..cov_ME5, M_E_avg, M_E_avg_raw
  data/processed/product_function_scores.csv        (200 x 25)
      product_id, n_reviews,
      MF1..MF7 (raw), MF1_z..MF7_z, cov_MF1..cov_MF7, M_F_avg, M_F_avg_raw
  data/processed/product_M_regression.csv           (200 x main-regression-ready)
      kansei + function + Y variables (if satisfaction exists)
  data/processed/kansei_function_aggregation_metadata.json

Aggregation rules:
  Affective (Kansei): final_value = (-1 if negated else 1) * polarity * degree_weight
  Functional:         final_value = polarity * degree_weight  (no negation reversal)
  Comment-level: mean(final_value) within the same comment+dimension
  Product-level Plan B: sum(comment_scores, missing=0) / N_total_reviews
  Coverage: n_mentioning_reviews / N_total_reviews
  z-score: per-dimension population z-score: (x - mean) / std (ddof=0)
  Aggregate: M_E_avg = mean(ME1_z..ME5_z); M_F_avg = mean(MF1_z..MF7_z)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def p(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = PROJECT_ROOT / "data" / "input" / "product_reviews_clean.xlsx"
EXTRACTION_DIR = PROJECT_ROOT / "data" / "extraction"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KANSEI_PARQUET = EXTRACTION_DIR / "llm_kansei_extractions.parquet"
FUNCTION_PARQUET = EXTRACTION_DIR / "llm_function_extractions.parquet"
DEGREE_JSON = PROJECT_ROOT / "lexicons" / "degree_words.json"

KANSEI_OUT = PROCESSED_DIR / "product_kansei_scores.csv"
FUNCTION_OUT = PROCESSED_DIR / "product_function_scores.csv"
COMBINED_OUT = PROCESSED_DIR / "product_M_regression.csv"
META_OUT = PROCESSED_DIR / "kansei_function_aggregation_metadata.json"

SATISFACTION_CSV = PROJECT_ROOT / "data" / "satisfaction" / "product_satisfaction.csv"

# Dimension -> column-name mapping (Stata-friendly).
# Affective (ME):
#   高级-廉价           -> ME1 Perceived Premiumness   (premium vs. cheap)
#   厚重-轻薄           -> ME2 Visual Solidity         (solid/substantial vs. flimsy)
#   简约大气-复杂凌乱   -> ME3 Minimalist Simplicity   (minimal/clean vs. cluttered)
#   精致美观-粗糙难看   -> ME4 Aesthetic Refinement    (refined/beautiful vs. rough/ugly)
#   时尚科技-传统老气   -> ME5 Tech-Modernity          (tech-modern vs. old-fashioned)
KANSEI_DIM_MAP = {
    "高级-廉价": "ME1",
    "厚重-轻薄": "ME2",
    "简约大气-复杂凌乱": "ME3",
    "精致美观-粗糙难看": "ME4",
    "时尚科技-传统老气": "ME5",
}
# Functional (MF):
#   升降功能      -> MF1 Lifting Performance
#   噪音控制      -> MF2 Noise Control
#   稳定性承重    -> MF3 Load-Bearing Stability
#   材质做工      -> MF4 Material and Build Quality
#   异味环保      -> MF5 Odor and Environmental Safety
#   操控智能      -> MF6 Control and Interface Integration
#   空间尺寸      -> MF7 Workspace Size Adequacy
FUNCTION_DIM_MAP = {
    "升降功能": "MF1",
    "噪音控制": "MF2",
    "稳定性承重": "MF3",
    "材质做工": "MF4",
    "异味环保": "MF5",
    "操控智能": "MF6",
    "空间尺寸": "MF7",
}

DEFAULT_WEIGHT_NO_DEGREE = 0.8
DEFAULT_WEIGHT_UNKNOWN = 0.8


# ============================================================
# Helpers
# ============================================================
def load_degree_lookup():
    with open(DEGREE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("weight_lookup", {})


def degree_to_weight(degree_word, lookup):
    if degree_word is None or (isinstance(degree_word, float) and np.isnan(degree_word)):
        return DEFAULT_WEIGHT_NO_DEGREE
    w = str(degree_word).strip()
    if not w:
        return DEFAULT_WEIGHT_NO_DEGREE
    v = lookup.get(w, None)
    if v is None:
        return DEFAULT_WEIGHT_UNKNOWN
    try:
        return float(v)
    except Exception:
        return DEFAULT_WEIGHT_UNKNOWN


def aggregate_dim_scores(matches_df, prod_n, dim_map, score_col_prefix,
                         apply_negation_reversal):
    """
    matches_df: long-format extraction with columns [product_id, commentId, dimension, polarity, degree_word, negated]
    prod_n:     DataFrame [product_id, N_total]
    dim_map:    {Chinese dimension name: column code}
    score_col_prefix: "ME" or "MF"
    apply_negation_reversal: True for affective, False for functional

    Returns:
        wide_score: pivoted 200 x len(dim_map) (column names = score_col_prefix1..N)
        wide_cov:   pivoted 200 x len(dim_map) (column names = cov_{prefix}1..N)
        diagnostic stats
    """
    df = matches_df.copy()
    # final_value
    df["degree_weight"] = df["degree_word"].apply(
        lambda w: degree_to_weight(w, DEGREE_LOOKUP_GLOBAL)
    )
    if apply_negation_reversal:
        df["sign"] = df["negated"].apply(lambda x: -1 if bool(x) else 1)
        df["final_value"] = df["sign"] * df["polarity"].astype(float) * df["degree_weight"]
    else:
        df["final_value"] = df["polarity"].astype(float) * df["degree_weight"]

    # Comment-level: mean within the same product+comment+dimension
    comment_dim = df.groupby(["product_id", "commentId", "dimension"], as_index=False
                             )["final_value"].mean()
    # Product-level sum
    prod_sum = comment_dim.groupby(["product_id", "dimension"], as_index=False
                                    )["final_value"].sum()
    prod_sum = prod_sum.merge(prod_n, on="product_id", how="left")
    prod_sum["product_score"] = prod_sum["final_value"] / prod_sum["N_total"]

    # Coverage: number of distinct mentioning reviews / total reviews
    cov = df.groupby(["product_id", "dimension"], as_index=False)["commentId"].nunique()
    cov = cov.rename(columns={"commentId": "n_mentioning"})
    cov = cov.merge(prod_n, on="product_id", how="left")
    cov["coverage"] = cov["n_mentioning"] / cov["N_total"]

    # Pivot to wide format
    wide_score = prod_sum.pivot_table(index="product_id", columns="dimension",
                                       values="product_score", fill_value=0).reset_index()
    wide_cov = cov.pivot_table(index="product_id", columns="dimension",
                                values="coverage", fill_value=0).reset_index()

    # Rename columns from Chinese to ME1/MF1 codes
    rename_score = {k: dim_map[k] for k in wide_score.columns if k in dim_map}
    rename_cov = {k: f"cov_{dim_map[k]}" for k in wide_cov.columns if k in dim_map}
    wide_score = wide_score.rename(columns=rename_score)
    wide_cov = wide_cov.rename(columns=rename_cov)

    return wide_score, wide_cov, df


def add_zscore_and_aggregate(df, raw_cols, family_letter):
    """
    For each raw_col, add a z-score column to df, and add two aggregate columns:
        M_{family_letter}_avg     = mean of z-scored cols (for the main analysis)
        M_{family_letter}_avg_raw = mean of raw cols (for descriptive statistics)
    Returns: (extended df, z_stats_dict, agg_col, raw_agg_col)
    """
    z_cols = []
    z_stats = {}
    for col in raw_cols:
        mu = float(df[col].mean())
        sd = float(df[col].std(ddof=0))
        if sd == 0:
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mu) / sd
        z_cols.append(f"{col}_z")
        z_stats[col] = {"mean_pop": mu, "std_pop": sd}

    agg_col = f"M_{family_letter}_avg"
    raw_agg_col = f"M_{family_letter}_avg_raw"
    df[agg_col] = df[z_cols].mean(axis=1)
    df[raw_agg_col] = df[raw_cols].mean(axis=1)

    return df, z_stats, agg_col, raw_agg_col


# ============================================================
# Main pipeline
# ============================================================
def main():
    p("=" * 64)
    p("  产品级 ME/MF 聚合 (方案 B: 未命中=0, 分母=总评论数)")
    p("=" * 64)

    # ---- 1. Load degree lookup ----
    p(f"\n[1/5] 加载 degree 查表 ({DEGREE_JSON.name})")
    global DEGREE_LOOKUP_GLOBAL
    DEGREE_LOOKUP_GLOBAL = load_degree_lookup()
    p(f"  词条数: {len(DEGREE_LOOKUP_GLOBAL)} (含 _default)")

    # ---- 2. Load total review count per product ----
    p(f"\n[2/5] 加载 {INPUT_XLSX.name} 计算 N_total per product")
    df_all = pd.read_excel(INPUT_XLSX, dtype=str)
    df_all["product_id"] = df_all["excel文件名"].str.extract(r"^(\d+-\d+)")
    prod_n = (df_all.groupby("product_id", sort=False).size()
              .rename("N_total").reset_index())
    prod_order = (df_all.drop_duplicates("product_id", keep="first")
                  [["product_id"]].reset_index(drop=True))
    prod_order["_ord"] = prod_order.index
    p(f"  产品数: {len(prod_order)}, 总评论数: {prod_n['N_total'].sum():,}")
    p(f"  评论数 min/median/max: {prod_n['N_total'].min()} / "
      f"{prod_n['N_total'].median():.0f} / {prod_n['N_total'].max()}")

    # ---- 3. Process affective (ME) ----
    p(f"\n[3/5] 聚合感性 ME (5 维)")
    if not KANSEI_PARQUET.exists():
        p(f"ERROR: 找不到 {KANSEI_PARQUET}, 请先跑 04_llm_extract_kansei.py")
        sys.exit(1)
    ka = pd.read_parquet(KANSEI_PARQUET)
    p(f"  感性 matches 总数: {len(ka):,}")
    p(f"  涉及产品数: {ka['product_id'].nunique()}")

    ka_score, ka_cov, ka_with_fv = aggregate_dim_scores(
        ka, prod_n, KANSEI_DIM_MAP, "ME", apply_negation_reversal=True
    )

    # Merge + fill missing dimension columns with 0
    ka_out = prod_order.merge(prod_n, on="product_id", how="left")
    ka_out = ka_out.merge(ka_score, on="product_id", how="left")
    ka_out = ka_out.merge(ka_cov, on="product_id", how="left")
    for me_label in [f"ME{i}" for i in range(1, 6)]:
        if me_label not in ka_out.columns:
            ka_out[me_label] = 0.0
        if f"cov_{me_label}" not in ka_out.columns:
            ka_out[f"cov_{me_label}"] = 0.0
    ka_out = ka_out.fillna(0)
    ka_out = ka_out.sort_values("_ord").drop(columns=["_ord"])
    ka_out = ka_out.rename(columns={"N_total": "n_reviews"})

    # Add z-score columns + M_E_avg (mean of z-scores) + M_E_avg_raw (mean of raw)
    me_raw_cols = [f"ME{i}" for i in range(1, 6)]
    ka_out, ka_z_stats, ka_agg_col, ka_agg_raw_col = add_zscore_and_aggregate(
        ka_out, me_raw_cols, family_letter="E"
    )

    me_cols = ["product_id", "n_reviews"] + \
              me_raw_cols + \
              [f"ME{i}_z" for i in range(1, 6)] + \
              [f"cov_ME{i}" for i in range(1, 6)] + \
              [ka_agg_col, ka_agg_raw_col]
    ka_out = ka_out[me_cols]
    ka_out.to_csv(KANSEI_OUT, index=False, encoding="utf-8-sig")
    p(f"  → {KANSEI_OUT.name}  shape={ka_out.shape}")
    p(f"     新增列: ME1_z..ME5_z, M_E_avg, M_E_avg_raw")

    # ---- 4. Process functional (MF) ----
    p(f"\n[4/5] 聚合功能 MF (7 维)")
    if not FUNCTION_PARQUET.exists():
        p(f"ERROR: 找不到 {FUNCTION_PARQUET}, 请先跑 05_llm_extract_function.py")
        sys.exit(1)
    fn = pd.read_parquet(FUNCTION_PARQUET)
    p(f"  功能 matches 总数: {len(fn):,}")
    p(f"  涉及产品数: {fn['product_id'].nunique()}")

    fn_score, fn_cov, fn_with_fv = aggregate_dim_scores(
        fn, prod_n, FUNCTION_DIM_MAP, "MF", apply_negation_reversal=False
    )

    fn_out = prod_order.merge(prod_n, on="product_id", how="left")
    fn_out = fn_out.merge(fn_score, on="product_id", how="left")
    fn_out = fn_out.merge(fn_cov, on="product_id", how="left")
    for mf_label in [f"MF{i}" for i in range(1, 8)]:
        if mf_label not in fn_out.columns:
            fn_out[mf_label] = 0.0
        if f"cov_{mf_label}" not in fn_out.columns:
            fn_out[f"cov_{mf_label}"] = 0.0
    fn_out = fn_out.fillna(0)
    fn_out = fn_out.sort_values("_ord").drop(columns=["_ord"])
    fn_out = fn_out.rename(columns={"N_total": "n_reviews"})

    # Add z-score columns + M_F_avg + M_F_avg_raw
    mf_raw_cols = [f"MF{i}" for i in range(1, 8)]
    fn_out, fn_z_stats, fn_agg_col, fn_agg_raw_col = add_zscore_and_aggregate(
        fn_out, mf_raw_cols, family_letter="F"
    )

    mf_cols = ["product_id", "n_reviews"] + \
              mf_raw_cols + \
              [f"MF{i}_z" for i in range(1, 8)] + \
              [f"cov_MF{i}" for i in range(1, 8)] + \
              [fn_agg_col, fn_agg_raw_col]
    fn_out = fn_out[mf_cols]
    fn_out.to_csv(FUNCTION_OUT, index=False, encoding="utf-8-sig")
    p(f"  → {FUNCTION_OUT.name}  shape={fn_out.shape}")
    p(f"     新增列: MF1_z..MF7_z, M_F_avg, M_F_avg_raw")

    # ---- 4.5 Merge kansei + function (+ Y if available) -> main-regression-ready file ----
    p(f"\n[4.5/5] 合并 kansei + function (+ Y if 可用) → {COMBINED_OUT.name}")
    # ka_out and fn_out both contain product_id, n_reviews, and their respective dimensions;
    # merge on product_id.
    combined = ka_out.drop(columns=["n_reviews"]).merge(
        fn_out, on="product_id", how="outer"
    )

    # Try to merge Y variables
    has_Y = False
    if SATISFACTION_CSV.exists():
        sat = pd.read_csv(SATISFACTION_CSV)
        keep_y = [c for c in [
            "product_id", "brand", "price",
            "sentiment_mean", "favorable_rate_raw", "favorable_rate_logit",
            "sentiment_z", "favrate_z",
            "Y_sentiment_only", "Y_favrate_only",
            "Y_50_50", "Y_30_70", "Y_70_30",
        ] if c in sat.columns]
        sat = sat[keep_y]
        combined = combined.merge(sat, on="product_id", how="left")
        has_Y = True
        p(f"     已合并满意度变量 ({len(keep_y) - 1} 列)")
    else:
        p(f"     WARN: 找不到 {SATISFACTION_CSV.name}, 跳过 Y 合并")

    # Sort by original product order
    combined = combined.merge(prod_order, on="product_id", how="left")
    combined = combined.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    combined.to_csv(COMBINED_OUT, index=False, encoding="utf-8-sig")
    p(f"  → {COMBINED_OUT.name}  shape={combined.shape}")

    # ---- 5. Metadata ----
    p(f"\n[5/5] 写元数据")

    me_stats = {}
    for me in [f"ME{i}" for i in range(1, 6)]:
        me_stats[me] = {
            "mean": float(ka_out[me].mean()),
            "std": float(ka_out[me].std()),
            "min": float(ka_out[me].min()),
            "max": float(ka_out[me].max()),
            "cov_mean": float(ka_out[f"cov_{me}"].mean()),
        }
    mf_stats = {}
    for mf in [f"MF{i}" for i in range(1, 8)]:
        mf_stats[mf] = {
            "mean": float(fn_out[mf].mean()),
            "std": float(fn_out[mf].std()),
            "min": float(fn_out[mf].min()),
            "max": float(fn_out[mf].max()),
            "cov_mean": float(fn_out[f"cov_{mf}"].mean()),
        }

    # Key diagnostic correlations (pre-regression sanity check)
    diag_corr = {
        "M_E_avg_vs_M_F_avg": float(combined["M_E_avg"].corr(combined["M_F_avg"])),
        "M_E_avg_raw_vs_M_F_avg_raw": float(combined["M_E_avg_raw"].corr(combined["M_F_avg_raw"])),
    }
    if has_Y:
        for y_col in ["Y_50_50", "Y_30_70", "Y_70_30",
                      "Y_sentiment_only", "Y_favrate_only"]:
            if y_col in combined.columns:
                diag_corr[f"M_E_avg_vs_{y_col}"] = float(combined["M_E_avg"].corr(combined[y_col]))
                diag_corr[f"M_F_avg_vs_{y_col}"] = float(combined["M_F_avg"].corr(combined[y_col]))

    meta = {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_products": int(len(ka_out)),
        "n_kansei_matches_total": int(len(ka)),
        "n_function_matches_total": int(len(fn)),
        "scoring_rules": {
            "kansei_final_value": "(-1 if negated else 1) * polarity * degree_weight",
            "function_final_value": "polarity * degree_weight  (no negation reversal)",
            "comment_level": "mean of final_values within same comment+dimension",
            "product_level": "Plan B: sum(comment_scores, missing=0) / N_total_reviews",
            "coverage": "n_mentioning_reviews / N_total_reviews",
            "z_score": "per-dimension population z-score: (x - mean) / std (ddof=0)",
            "M_E_avg": "mean of ME1_z..ME5_z (recommended for the main analysis, 5 dims equal-weighted)",
            "M_F_avg": "mean of MF1_z..MF7_z",
            "M_E_avg_raw": "mean of ME1..ME5 raw (for descriptive statistics)",
            "M_F_avg_raw": "mean of MF1..MF7 raw",
            "degree_lookup_source": str(DEGREE_JSON),
            "default_weight_no_degree": DEFAULT_WEIGHT_NO_DEGREE,
            "default_weight_unknown": DEFAULT_WEIGHT_UNKNOWN,
        },
        "dimension_mapping": {
            "kansei": KANSEI_DIM_MAP,
            "function": FUNCTION_DIM_MAP,
        },
        "kansei_score_distribution": me_stats,
        "function_score_distribution": mf_stats,
        "z_score_population_stats": {
            "kansei": ka_z_stats,
            "function": fn_z_stats,
        },
        "diagnostic_correlations": diag_corr,
        "output_files": {
            "kansei": str(KANSEI_OUT),
            "function": str(FUNCTION_OUT),
            "combined_regression_ready": str(COMBINED_OUT),
        },
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    p(f"  → {META_OUT.name}")

    # ---- Wrap up ----
    p("\n" + "=" * 64)
    p("  完成")
    p("=" * 64)
    p("\n感性 ME 描述统计:")
    p(f"  {'维度':<8} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'cov':>8}")
    for me in [f"ME{i}" for i in range(1, 6)]:
        s = me_stats[me]
        p(f"  {me:<8} {s['mean']:+10.4f} {s['std']:10.4f} "
          f"{s['min']:+10.4f} {s['max']:+10.4f} {s['cov_mean']:8.3f}")

    p("\n功能 MF 描述统计:")
    p(f"  {'维度':<8} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'cov':>8}")
    for mf in [f"MF{i}" for i in range(1, 8)]:
        s = mf_stats[mf]
        p(f"  {mf:<8} {s['mean']:+10.4f} {s['std']:10.4f} "
          f"{s['min']:+10.4f} {s['max']:+10.4f} {s['cov_mean']:8.3f}")

    p(f"\n汇总维度 (z-score 平均):")
    p(f"  M_E_avg     mean={combined['M_E_avg'].mean():+.4f}  std={combined['M_E_avg'].std():.4f}  "
      f"range=[{combined['M_E_avg'].min():+.3f}, {combined['M_E_avg'].max():+.3f}]")
    p(f"  M_F_avg     mean={combined['M_F_avg'].mean():+.4f}  std={combined['M_F_avg'].std():.4f}  "
      f"range=[{combined['M_F_avg'].min():+.3f}, {combined['M_F_avg'].max():+.3f}]")
    p(f"  M_E_avg_raw mean={combined['M_E_avg_raw'].mean():+.4f}  std={combined['M_E_avg_raw'].std():.4f}")
    p(f"  M_F_avg_raw mean={combined['M_F_avg_raw'].mean():+.4f}  std={combined['M_F_avg_raw'].std():.4f}")

    p(f"\n关键诊断相关性 (回归前 sanity check):")
    p(f"  M_E_avg ↔ M_F_avg:           r = {diag_corr['M_E_avg_vs_M_F_avg']:+.4f}")
    if has_Y:
        p(f"  ───── 与综合满意度 Y_50_50 ─────")
        p(f"  M_E_avg ↔ Y_50_50:           r = {diag_corr.get('M_E_avg_vs_Y_50_50', float('nan')):+.4f}  ← 感性路径")
        p(f"  M_F_avg ↔ Y_50_50:           r = {diag_corr.get('M_F_avg_vs_Y_50_50', float('nan')):+.4f}  ← 功能路径")
        p(f"  ───── 与单独 LLM 情感 ─────")
        p(f"  M_E_avg ↔ Y_sentiment_only:  r = {diag_corr.get('M_E_avg_vs_Y_sentiment_only', float('nan')):+.4f}")
        p(f"  M_F_avg ↔ Y_sentiment_only:  r = {diag_corr.get('M_F_avg_vs_Y_sentiment_only', float('nan')):+.4f}")
        p(f"  ───── 与单独好评率(logit) ─────")
        p(f"  M_E_avg ↔ Y_favrate_only:    r = {diag_corr.get('M_E_avg_vs_Y_favrate_only', float('nan')):+.4f}")
        p(f"  M_F_avg ↔ Y_favrate_only:    r = {diag_corr.get('M_F_avg_vs_Y_favrate_only', float('nan')):+.4f}")

    p(f"\n输出文件:")
    p(f"  {KANSEI_OUT}")
    p(f"  {FUNCTION_OUT}")
    p(f"  {COMBINED_OUT}   ← 主回归就绪文件")
    p(f"  {META_OUT}")
    p(f"\n下一步: 视觉特征 PCA + 三层融合, 然后进入 Stata 主回归")
    p(f"  Stata 主分析建议: reg Y_50_50 M_E_avg M_F_avg [控制变量], vce(robust)")


if __name__ == "__main__":
    main()
