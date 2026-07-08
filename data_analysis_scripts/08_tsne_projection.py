#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_tsne_projection.py
=====================

Creates t-SNE projection plots for the fingerprint space.

Quality metrics:
- Trustworthiness@10
- Normalized Kruskal stress

Output folder:
  analysis_outputs/08_tsne_projection/
"""

import inspect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

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
PERPLEXITY = 30
LEARNING_RATE = 200
MAX_ITER = 1000

OUTDIR = "analysis_outputs/08_tsne_projection"


def normalized_kruskal_stress(X_high, X_low):
    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low = pairwise_distances(X_low, metric="euclidean")
    den = np.sum(D_high ** 2)
    if den <= 0:
        return np.nan
    return float(np.sqrt(np.sum((D_high - D_low) ** 2) / den))


def run_tsne(X):
    kwargs = dict(
        n_components=2,
        perplexity=PERPLEXITY,
        learning_rate=LEARNING_RATE,
        init="pca",
        random_state=RANDOM_STATE,
    )
    sig = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in sig:
        kwargs["max_iter"] = MAX_ITER
    else:
        kwargs["n_iter"] = MAX_ITER
    return TSNE(**kwargs).fit_transform(X)


def plot_class(X2, y, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1], s=18, alpha=0.65, label=f"class {cls}")
    ax.legend(title="Class")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_continuous(X2, values, title, label, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=values, s=18, alpha=0.70)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(label)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    rf = make_rf(random_state=RANDOM_STATE, n_estimators=300)
    rf.fit(X_train, y_train)
    prob = positive_scores(rf, X)

    rng = np.random.RandomState(RANDOM_STATE)
    if X_scaled.shape[0] > MAX_POINTS:
        sample_idx = rng.choice(X_scaled.shape[0], size=MAX_POINTS, replace=False)
    else:
        sample_idx = np.arange(X_scaled.shape[0])

    Xs = X_scaled[sample_idx]
    ys = y[sample_idx]
    ps = prob[sample_idx]

    X2 = run_tsne(Xs)
    T = trustworthiness(Xs, X2, n_neighbors=10)
    stress = normalized_kruskal_stress(Xs, X2)

    pd.DataFrame({
        "tsne1": X2[:, 0],
        "tsne2": X2[:, 1],
        "y": ys,
        "rf_probability_class1": ps,
    }).to_csv(outdir / "tsne_coordinates.csv", index=False)

    pd.DataFrame([{
        "method": "t-SNE",
        "trustworthiness@10": T,
        "normalized_kruskal_stress": stress,
        "n_points": len(sample_idx),
    }]).to_csv(outdir / "tsne_quality_metrics.csv", index=False)

    title = f"t-SNE projection of fingerprint space\nTrustworthiness@10 = {T:.3f} | Normalized Kruskal stress = {stress:.3f}"
    plot_class(X2, ys, title + " | colored by class", outdir / "tsne_projection_colored_by_class.png")
    plot_continuous(X2, ps, title + " | colored by RF probability", "RF probability class 1", outdir / "tsne_projection_colored_by_rf_probability.png")

    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
