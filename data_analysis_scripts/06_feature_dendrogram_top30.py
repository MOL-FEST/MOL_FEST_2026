#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_feature_dendrogram_top30.py
==============================

Creates a hierarchical clustering dendrogram for the top ANOVA features.

Distance used:
  distance = 1 - Pearson correlation

Output folder:
  analysis_outputs/06_feature_dendrogram/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

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
OUTDIR = "analysis_outputs/06_feature_dendrogram"


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
    X_top = pd.DataFrame(X[:, top_idx], columns=top_features)

    corr = X_top.corr(method="pearson").fillna(0.0)
    corr.to_csv(outdir / f"top{len(top_features)}_feature_correlation_matrix.csv")

    distance = 1.0 - corr.values
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)

    Z = linkage(squareform(distance, checks=False), method="average")

    plt.figure(figsize=(16, 8))
    dendrogram(
        Z,
        labels=top_features,
        leaf_rotation=90,
        leaf_font_size=9,
    )
    plt.ylabel("Distance (1 - correlation)")
    plt.xlabel("Features")
    plt.title(f"Hierarchical Clustering of Top {len(top_features)} Features")
    plt.tight_layout()
    plt.savefig(outdir / f"top{len(top_features)}_feature_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
