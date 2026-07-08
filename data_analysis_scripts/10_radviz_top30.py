#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_radviz_top30.py
==================

Creates a RadViz plot using the top ANOVA features.

Output folder:
  analysis_outputs/10_radviz/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pandas.plotting import radviz
from common_data_features import load_features_from_csv, top_anova_features, ensure_dir

# =========================
# USER SETTINGS
# =========================
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"
TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"

FEATURE_MODE = "smiles"
FINGERPRINT_TYPE = "maccs"
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
POSITIVE_THRESHOLD = None

TOP_N = 30
OUTDIR = "analysis_outputs/10_radviz"


def main():
    outdir = ensure_dir(OUTDIR)

    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    scores, top_features = top_anova_features(X, y, feature_names, TOP_N)
    scores.to_csv(outdir / "anova_f_score_all_features.csv", index=False)

    top_idx = [feature_names.index(f) for f in top_features]
    rv_df = pd.DataFrame(X[:, top_idx], columns=top_features)
    rv_df["Class"] = pd.Series(y).map({0: "class_0", 1: "class_1"}).values

    plt.figure(figsize=(12, 9))
    ax = radviz(rv_df, "Class", alpha=0.65, s=22)
    ax.set_title(f"RadViz — classes: {len(set(y))} | features: {len(top_features)}")
    plt.tight_layout()
    plt.savefig(outdir / f"radviz_top{len(top_features)}_anova_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
