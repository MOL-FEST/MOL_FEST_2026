#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_correlation_heatmap_top30.py
===============================

Creates a Pearson correlation heatmap for the top ANOVA features plus the target.

Output folder:
  analysis_outputs/05_correlation_heatmap/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUTDIR = "analysis_outputs/05_correlation_heatmap"


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
    plot_df = pd.DataFrame(X[:, top_idx], columns=top_features)
    plot_df["Target_class1"] = y

    corr = plot_df.corr(method="pearson").fillna(0.0)
    corr.to_csv(outdir / f"top{len(top_features)}_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(f"Correlation Heatmap — Top {len(top_features)} ANOVA Features + Target")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    plt.tight_layout()
    plt.savefig(outdir / f"top{len(top_features)}_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
