#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_pca_projection.py
====================

Creates PCA projection plots for the fingerprint space:
1) colored by observed class
2) colored by RF class-1 probability

Output folder:
  analysis_outputs/07_pca_projection/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from common_data_features import load_features_from_csv, make_rf, positive_scores, ensure_dir

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

RANDOM_STATE = 42
OUTDIR = "analysis_outputs/07_pca_projection"


def plot_class(X2, y, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1], s=18, alpha=0.65, label=f"class {cls}")
    ax.legend(title="Class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_continuous(X2, values, title, label, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=values, s=18, alpha=0.70)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    outdir = ensure_dir(OUTDIR)

    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    rf = make_rf(random_state=RANDOM_STATE, n_estimators=300)
    rf.fit(X_train, y_train)
    prob = positive_scores(rf, X)

    coords = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "y": y,
        "rf_probability_class1": prob,
    })
    coords.to_csv(outdir / "pca_coordinates.csv", index=False)

    title_base = f"PCA projection of fingerprint space\nExplained variance: PC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%"
    plot_class(X2, y, title_base + " | colored by class", outdir / "pca_projection_colored_by_class.png")
    plot_continuous(X2, prob, title_base + " | colored by RF probability", "RF probability class 1", outdir / "pca_projection_colored_by_rf_probability.png")

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
