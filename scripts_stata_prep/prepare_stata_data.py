"""
prepare_stata_data.py
=========================
Normalize product_id across all CSVs and merge them into a single Stata .dta file.
This avoids problems where Stata's `import delimited` mishandles product_id values
that contain special characters.

Inputs:
  data/processed/product_M_regression.csv
  data/processed/product_visual_pcs.csv
  data/processed/product_kansei_scores.csv
  data/processed/product_function_scores.csv
  data/satisfaction/product_satisfaction.csv

Output:
  data/processed/product_master.dta   -- loadable in Stata via `use`
"""

import sys
from pathlib import Path
import pandas as pd


def p(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def normalize_pid(s):
    """Normalize product_id formatting: regular hyphen, no whitespace."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return (str(s).strip()
            .replace("—", "-").replace("–", "-")
            .replace("－", "-").replace("‐", "-")
            .replace("　", "").replace(" ", ""))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data/processed"
SAT = PROJECT_ROOT / "data/satisfaction"

OUT_DTA = PROC / "product_master.dta"
OUT_CSV = PROC / "product_master.csv"


def main():
    p("=" * 64)
    p("  生成 Stata 主数据 product_master.dta")
    p("=" * 64)

    # 1. M_regression (main table: Y + 5 ME + 7 MF + ME_z/MF_z + M_E_avg/M_F_avg + satisfaction Y)
    p("\n[1] 加载 product_M_regression.csv")
    m = pd.read_csv(PROC / "product_M_regression.csv", dtype=str)
    m["product_id"] = m["product_id"].apply(normalize_pid)
    p(f"  shape: {m.shape}")

    # Convert numeric columns to float
    keep_str = ["product_id", "brand"]
    for c in m.columns:
        if c not in keep_str:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    # 2. visual PCs
    p("\n[2] 加载 product_visual_pcs.csv")
    pcs = pd.read_csv(PROC / "product_visual_pcs.csv", dtype=str)
    pcs["product_id"] = pcs["product_id"].apply(normalize_pid)
    for c in pcs.columns:
        if c != "product_id":
            pcs[c] = pd.to_numeric(pcs[c], errors="coerce")
    p(f"  shape: {pcs.shape}, cols: {pcs.columns.tolist()}")

    # 3. satisfaction (brand, price, n_reviews_total)
    p("\n[3] 加载 satisfaction/product_satisfaction.csv")
    sat = pd.read_csv(SAT / "product_satisfaction.csv", dtype=str)
    sat["product_id"] = sat["product_id"].apply(normalize_pid)
    for c in ["price", "n_reviews_total"]:
        if c in sat.columns:
            sat[c] = pd.to_numeric(sat[c], errors="coerce")
    sat_keep = sat[["product_id", "brand", "price", "n_reviews_total"]].copy() \
        if "brand" in sat.columns else sat[["product_id", "price", "n_reviews_total"]].copy()
    p(f"  保留列: {sat_keep.columns.tolist()}")

    # ME / MF dimensions are already in product_M_regression.csv -- no need to load them again
    p("\n[4-5] ME1_z..ME5_z 和 MF1_z..MF7_z 已在 M_regression 中(跳过单独加载)")

    # ---- Merge ----
    p("\n[6] 合并所有数据集")
    combined = m.copy()

    pcs_cols = ["product_id"] + [c for c in pcs.columns if c.startswith("z_pc")]
    combined = combined.merge(pcs[pcs_cols], on="product_id", how="left")
    p(f"  + visual PCs: {combined.shape}")

    # If m already has brand/price columns, drop them first
    for col in ["brand", "price", "n_reviews_total"]:
        if col in combined.columns and col in sat_keep.columns:
            combined = combined.drop(columns=[col])
    combined = combined.merge(sat_keep, on="product_id", how="left")
    p(f"  + brand/price/n_reviews: {combined.shape}")

    # ME/MF already in m; skip

    # ---- Sanity check ----
    p("\n[7] 数据完整性检查")
    p(f"  总行数: {len(combined)}")
    p(f"  唯一 product_id: {combined['product_id'].nunique()}")
    p(f"  product_id 样例: {combined['product_id'].head(5).tolist()}")

    # NaN check on critical columns
    critical = ["Y_50_50", "M_E_avg", "M_F_avg", "z_pc1", "z_pc7", "price", "n_reviews_total"]
    for c in critical:
        if c in combined.columns:
            nn = combined[c].isna().sum()
            mark = " ⚠️" if nn > 0 else ""
            p(f"  {c:<20s}  NaN={nn}{mark}")
        else:
            p(f"  {c:<20s}  ✗ 缺失列!")

    # Lowercase column names (Stata-friendly)
    combined.columns = [c.lower() for c in combined.columns]

    # Stata does not allow special characters in column names -- confirm there are none
    bad_cols = [c for c in combined.columns
                if not all(ch.isalnum() or ch == "_" for ch in c)]
    if bad_cols:
        p(f"  WARN 列名含特殊字符: {bad_cols}")
        # Force-clean
        combined.columns = ["".join(ch if (ch.isalnum() or ch == "_") else "_"
                                     for ch in c) for c in combined.columns]

    # ---- Export ----
    p(f"\n[8] 导出")
    # version=117 for Stata 13+, version=118 for Stata 14+
    try:
        combined.to_stata(OUT_DTA, write_index=False, version=118)
    except Exception as e:
        p(f"  WARN version=118 失败 ({e}), 降级到 117")
        combined.to_stata(OUT_DTA, write_index=False, version=117)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    p(f"  → {OUT_DTA}")
    p(f"  → {OUT_CSV}")

    p("\n" + "=" * 64)
    p(f"  完成. Stata 直接 use \"{OUT_DTA.as_posix()}\", clear")
    p("=" * 64)
    p(f"\n关键变量速查:")
    p(f"  Y:       y_50_50  (主满意度)")
    p(f"           y_30_70, y_70_30, y_sentiment_only, y_favrate_only  (替代)")
    p(f"  M_E:     m_e_avg")
    p(f"  M_F:     m_f_avg")
    p(f"  视觉 X:  z_pc1 - z_pc7")
    p(f"  分维度 ME: me1_z - me5_z")
    p(f"  分维度 MF: mf1_z - mf7_z")
    p(f"  控制:    price, n_reviews_total (回归前在 Stata 里 gen lnX)")
    p(f"  品牌:    brand")
    p(f"  主样本:  n_reviews_total >= 20  → N=185")


if __name__ == "__main__":
    main()
