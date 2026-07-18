"""
fdr_correction.py
==================
Benjamini-Hochberg FDR correction for the 12-mediator dimensional decomposition.

Input:  stata/output/T6_decomp_pvalues.csv     (raw p-values of 12 mediators)
Output: stata/output/T6_decomp_pvalues_fdr.csv (with FDR-adjusted p-values)

Primary outcome: platform favorable rate. Full seven-control specification.

Reference:
  Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
  JRSS-B, 57(1), 289-300.
"""

import sys
from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV  = PROJECT_ROOT / "stata/output/T6_decomp_pvalues.csv"
OUT_CSV = PROJECT_ROOT / "stata/output/T6_decomp_pvalues_fdr.csv"


def main():
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run stata/main_analysis.do first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} mediator p-values")

    reject_05, p_adj_05, _, _ = multipletests(df["p_raw"], alpha=0.05, method="fdr_bh")
    reject_10, p_adj_10, _, _ = multipletests(df["p_raw"], alpha=0.10, method="fdr_bh")

    df["p_fdr_bh"]     = p_adj_05
    df["sig_fdr_0.05"] = reject_05
    df["sig_fdr_0.10"] = reject_10
    df["stars_fdr"] = df.apply(
        lambda r: "***" if r["p_fdr_bh"] < 0.01
              else "**"  if r["p_fdr_bh"] < 0.05
              else "*"   if r["p_fdr_bh"] < 0.10
              else "ns",
        axis=1,
    )

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSignificant (p<0.05): {(df['p_raw']<0.05).sum()}/{len(df)}")
    print(f"Significant (q<0.05): {reject_05.sum()}/{len(df)}")
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
