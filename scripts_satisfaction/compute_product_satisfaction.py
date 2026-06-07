"""
compute_product_satisfaction.py
====================================
Compute product-level satisfaction Y (5 variants).

Inputs:
  data/satisfaction/comment_sentiment_scores.csv   (from scripts_extraction/llm_sentiment_scoring.py)
  product_reviews_clean.xlsx                       (favorable_rate + metadata)

Outputs:
  data/satisfaction/product_satisfaction.csv       (main table, 200 rows)
  data/satisfaction/product_satisfaction_metadata.json

Processing logic:
  1. Aggregate LLM sentiment scores per product -> sentiment_mean in [-1, 1]
  2. Apply logit transform (epsilon=0.001) to the platform favorable rate -> favorable_rate_logit
  3. z-score standardize each
  4. Five Y variants:
     - Y_sentiment_only  = sentiment_z
     - Y_favrate_only    = favrate_z
     - Y_50_50           = 0.5*sentiment_z + 0.5*favrate_z
     - Y_30_70           = 0.3*sentiment_z + 0.7*favrate_z
     - Y_70_30           = 0.7*sentiment_z + 0.3*favrate_z
  5. included_185: flag products with n_reviews_total >= 20 (main analysis sample)
  6. Output order matches the first-appearance order in the source Excel (1-1, 1-2, ..., 9-6)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = PROJECT_ROOT / "data" / "input" / "product_reviews_clean.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "satisfaction"

SENTIMENT_CSV = OUTPUT_DIR / "comment_sentiment_scores.csv"
PRODUCT_CSV = OUTPUT_DIR / "product_satisfaction.csv"
PRODUCT_META_JSON = OUTPUT_DIR / "product_satisfaction_metadata.json"

EPSILON = 0.001                # epsilon for the logit transform
MIN_REVIEWS_THRESHOLD = 20     # main analysis sample: n_reviews >= 20 -> N=185
POS_THRESHOLD = 0.1            # positive/negative count thresholds
NEG_THRESHOLD = -0.1


# ============================================================
# Helpers
# ============================================================
def logit_transform(p, eps=EPSILON):
    """logit(p) with epsilon smoothing. 0.85 -> 1.73, 1.00 -> 6.91."""
    p = np.asarray(p, dtype=float)
    return np.log((p + eps) / (1.0 - p + eps))


def safe_zscore(x):
    """z-score standardization (population std, ddof=0). Returns 0 if std=0."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x), float(mu), 0.0
    return (x - mu) / sd, float(mu), float(sd)


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("=" * 64)
    print("  产品级满意度变量计算")
    print("=" * 64)

    # ---------- 1. Load sentiment scores ----------
    print(f"\n[1/4] 加载 {SENTIMENT_CSV.name}")
    if not SENTIMENT_CSV.exists():
        raise FileNotFoundError(
            f"未找到 {SENTIMENT_CSV},请先运行 02_llm_sentiment_scoring.py"
        )
    sent = pd.read_csv(SENTIMENT_CSV)
    sent["sentiment_score"] = pd.to_numeric(sent["sentiment_score"], errors="coerce")
    n_total_comments = len(sent)
    n_valid_comments = int(sent["sentiment_score"].notna().sum())
    print(f"  评论数: {n_total_comments:,}")
    print(f"  有效情感分: {n_valid_comments:,}  ({n_valid_comments/n_total_comments*100:.2f}%)")

    # ---------- 2. Load source Excel for product-level metadata (preserving order) ----------
    print(f"\n[2/4] 加载 {INPUT_XLSX.name}")
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    df = df.reset_index(drop=True)
    df["product_id"] = df["excel文件名"].str.extract(r"^(\d+-\d+)")
    df["favorable_rate"] = pd.to_numeric(df["favorable rate"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # First-appearance order of products in the source table (used for final sorting)
    prod_order = (
        df.drop_duplicates("product_id", keep="first")[["product_id"]]
        .reset_index(drop=True)
        .rename(columns={"product_id": "product_id"})
    )
    prod_order["_prod_order"] = prod_order.index
    print(f"  唯一产品: {len(prod_order)}")

    # ---------- 3. Product-level aggregation ----------
    print(f"\n[3/4] 计算产品级变量")

    # 3.1 Raw platform favorable rate + metadata
    prod_meta = (
        df.groupby("product_id", sort=False)
        .agg(
            favorable_rate_raw=("favorable_rate", "first"),
            price=("price", "first"),
            brand=("brand", "first"),
            n_reviews_total=("commentId", "count"),
        )
        .reset_index()
    )

    # 3.2 Aggregate LLM sentiment scores
    sent_agg = (
        sent.groupby("product_id", sort=False)
        .agg(
            n_reviews_used=("sentiment_score", lambda x: int(x.notna().sum())),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            n_pos=("sentiment_score", lambda x: int((x > POS_THRESHOLD).sum())),
            n_neg=("sentiment_score", lambda x: int((x < NEG_THRESHOLD).sum())),
            n_neu=("sentiment_score",
                   lambda x: int(((x >= NEG_THRESHOLD) & (x <= POS_THRESHOLD)).sum())),
        )
        .reset_index()
    )

    # 3.3 Merge
    prod = prod_meta.merge(sent_agg, on="product_id", how="left")

    # 3.4 logit transform
    prod["favorable_rate_logit"] = logit_transform(prod["favorable_rate_raw"].values)

    # 3.5 z-score (population std)
    sent_z, sent_mu, sent_sd = safe_zscore(prod["sentiment_mean"].values)
    fav_z, fav_mu, fav_sd = safe_zscore(prod["favorable_rate_logit"].values)
    prod["sentiment_z"] = sent_z
    prod["favrate_z"] = fav_z

    # 3.6 Five Y variants
    prod["Y_sentiment_only"] = prod["sentiment_z"]
    prod["Y_favrate_only"] = prod["favrate_z"]
    prod["Y_50_50"] = 0.5 * prod["sentiment_z"] + 0.5 * prod["favrate_z"]
    prod["Y_30_70"] = 0.3 * prod["sentiment_z"] + 0.7 * prod["favrate_z"]
    prod["Y_70_30"] = 0.7 * prod["sentiment_z"] + 0.3 * prod["favrate_z"]

    # 3.7 Main analysis sample flag
    prod["included_185"] = prod["n_reviews_total"] >= MIN_REVIEWS_THRESHOLD

    # 3.8 Restore original product order
    prod = prod.merge(prod_order, on="product_id").sort_values("_prod_order").reset_index(drop=True)
    prod = prod.drop(columns=["_prod_order"])

    # ---------- 4. Output ----------
    print(f"\n[4/4] 保存输出")
    out_cols = [
        "product_id", "brand", "price",
        "n_reviews_total", "n_reviews_used",
        "n_pos", "n_neu", "n_neg",
        "sentiment_mean", "sentiment_std",
        "favorable_rate_raw", "favorable_rate_logit",
        "sentiment_z", "favrate_z",
        "Y_sentiment_only", "Y_favrate_only",
        "Y_50_50", "Y_30_70", "Y_70_30",
        "included_185",
    ]
    prod[out_cols].to_csv(PRODUCT_CSV, index=False, encoding="utf-8-sig")
    print(f"  写入 → {PRODUCT_CSV.name}")

    # Metadata
    corr = float(prod[["sentiment_z", "favrate_z"]].corr().iloc[0, 1])
    metadata = {
        "n_products": len(prod),
        "n_included_main": int(prod["included_185"].sum()),
        "n_excluded_main": int((~prod["included_185"]).sum()),
        "min_reviews_threshold": MIN_REVIEWS_THRESHOLD,
        "epsilon_logit": EPSILON,
        "thresholds_for_pos_neu_neg": {
            "positive": f"score > {POS_THRESHOLD}",
            "neutral":  f"{NEG_THRESHOLD} <= score <= {POS_THRESHOLD}",
            "negative": f"score < {NEG_THRESHOLD}",
        },
        "z_score_population_stats": {
            "sentiment_mean_pop": sent_mu,
            "sentiment_std_pop": sent_sd,
            "favorable_rate_logit_mean_pop": fav_mu,
            "favorable_rate_logit_std_pop": fav_sd,
        },
        "correlation_sentiment_z_vs_favrate_z": round(corr, 4),
        "Y_summary": {
            col: {
                "mean": float(prod[col].mean()),
                "std": float(prod[col].std(ddof=0)),
                "min": float(prod[col].min()),
                "max": float(prod[col].max()),
            }
            for col in ["Y_sentiment_only", "Y_favrate_only", "Y_50_50", "Y_30_70", "Y_70_30"]
        },
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    PRODUCT_META_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  写入 → {PRODUCT_META_JSON.name}")

    # ---------- 5. Final summary ----------
    print("\n" + "=" * 64)
    print("  完成")
    print("=" * 64)
    print(f"产品总数: {len(prod)}")
    print(f"主分析样本 (≥{MIN_REVIEWS_THRESHOLD} 评论): {prod['included_185'].sum()}")
    print(f"剔除产品: {(~prod['included_185']).sum()}")
    print()
    print(f"情感均值统计:  mean={sent_mu:+.4f}  std={sent_sd:.4f}")
    print(f"好评率 logit:  mean={fav_mu:+.4f}  std={fav_sd:.4f}")
    print(f"sentiment_z vs favrate_z 相关系数: {corr:+.4f}")
    print()
    print(f"输出主表: {PRODUCT_CSV}")
    print(f"元数据:   {PRODUCT_META_JSON}")
    print()
    print("Y 变体说明:")
    print("  Y_sentiment_only  = sentiment_z              (纯 LLM 情感)")
    print("  Y_favrate_only    = favrate_z                (纯平台好评率, logit)")
    print("  Y_50_50           = 0.5*sent + 0.5*fav       (主分析推荐)")
    print("  Y_30_70           = 0.3*sent + 0.7*fav       (偏好评率)")
    print("  Y_70_30           = 0.7*sent + 0.3*fav       (偏情感)")


if __name__ == "__main__":
    main()
