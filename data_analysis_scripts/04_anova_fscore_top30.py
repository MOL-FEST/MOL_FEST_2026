#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_anova_fscore_top30.py
========================

Creates a bar chart of the top features by ANOVA F-score for binary classification.

Output folder:
  analysis_outputs/04_anova_fscore/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_data_features import load_features_from_csv, top_anova_features, ensure_dir

# =========================
# USER SETTINGS
# =========================
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"
TARGET_COLUMN = "binary_label_agonist1_antagonist0"  # or "AUTO"
SMILES_COLUMN = "QSAR-Ready SMILES"                   # or "AUTO"

FEATURE_MODE = "smiles"        # "smiles", "existing", or "auto"
FINGERPRINT_TYPE = "maccs"     # "maccs" gives MACCS_49-like labels
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
POSITIVE_THRESHOLD = None

TOP_N = 30
OUTDIR = "analysis_outputs/04_anova_fscore"


def main():
    outdir = ensure_dir(OUTDIR)

    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    print(f"Input: {input_path}")
    print(f"Rows: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Target: {target_col}")

    scores, top_features = top_anova_features(X, y, feature_names, TOP_N)
    scores.to_csv(outdir / "anova_f_score_all_features.csv", index=False)

    top = scores.head(TOP_N)
    plt.figure(figsize=(18, 8))
    x = np.arange(len(top))
    plt.bar(x, top["F_score"].values)
    plt.xticks(x, top["Feature"].values, rotation=90)
    plt.xlabel("Features", fontweight="bold")
    plt.ylabel("F-score", fontweight="bold")
    plt.title(f"Top {TOP_N} Most Important Features Based on ANOVA F-score", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / f"anova_f_score_top{TOP_N}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(top[["Rank", "Feature", "F_score", "p_value"]].to_string(index=False))
    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
