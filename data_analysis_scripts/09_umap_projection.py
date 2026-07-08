#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_umap_projection.py
=====================

Creates UMAP projection plots for the fingerprint space.

Output folder:
  analysis_outputs/09_umap_projection/

Install:
  pip install umap-learn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import trustworthiness
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
MAX_POINTS = 2500
N_NEIGHBORS = 15
MIN_DIST = 0.1

OUTDIR = "analysis_outputs/09_umap_projection"


def is_binary_matrix(X):
    vals = np.unique(X[~np.isnan(X)])
    return len(vals) > 0 and set(vals.tolist()).issubset({0.0, 1.0})


def plot_class(X2, y, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1], s=18, alpha=0.65, label=f"class {cls}")
    ax.legend(title="Class")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_continuous(X2, values, title, label, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=values, s=18, alpha=0.70)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(label)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    try:
        import umap
    except ImportError as e:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn") from e

    outdir = ensure_dir(OUTDIR)

    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    rf = make_rf(random_state=RANDOM_STATE, n_estimators=300)
    rf.fit(X_train, y_train)
    prob = positive_scores(rf, X)

    rng = np.random.RandomState(RANDOM_STATE)
    if X.shape[0] > MAX_POINTS:
        sample_idx = rng.choice(X.shape[0], size=MAX_POINTS, replace=False)
    else:
        sample_idx = np.arange(X.shape[0])

    Xs_raw = X[sample_idx]
    ys = y[sample_idx]
    ps = prob[sample_idx]

    if is_binary_matrix(Xs_raw):
        X_umap_in = Xs_raw.astype(np.float32)
        metric = "jaccard"
        X_for_quality = X_umap_in
    else:
        X_umap_in = StandardScaler().fit_transform(Xs_raw)
        metric = "euclidean"
        X_for_quality = X_umap_in

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=metric,
        random_state=RANDOM_STATE,
    )
    X2 = reducer.fit_transform(X_umap_in)
    T = trustworthiness(X_for_quality, X2, n_neighbors=10)

    pd.DataFrame({
        "umap1": X2[:, 0],
        "umap2": X2[:, 1],
        "y": ys,
        "rf_probability_class1": ps,
    }).to_csv(outdir / "umap_coordinates.csv", index=False)

    pd.DataFrame([{
        "method": "UMAP",
        "metric": metric,
        "trustworthiness@10": T,
        "n_points": len(sample_idx),
    }]).to_csv(outdir / "umap_quality_metrics.csv", index=False)

    title = f"UMAP projection of fingerprint space | metric={metric}\nTrustworthiness@10 = {T:.3f}"
    plot_class(X2, ys, title + " | colored by class", outdir / "umap_projection_colored_by_class.png")
    plot_continuous(X2, ps, title + " | colored by RF probability", "RF probability class 1", outdir / "umap_projection_colored_by_rf_probability.png")

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
