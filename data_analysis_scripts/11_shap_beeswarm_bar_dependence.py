#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_shap_beeswarm_bar_dependence.py
==================================

Trains a Random Forest classifier and creates:
- SHAP beeswarm plot
- SHAP bar plot
- SHAP dependence plots for top features
- top SHAP feature table

Output folder:
  analysis_outputs/11_shap_beeswarm/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score

from common_data_features import (
    load_features_from_csv, make_rf, positive_scores, extract_class1_shap,
    ensure_dir, safe_filename
)

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
TEST_SIZE = 0.20
RF_N_ESTIMATORS = 500

SHAP_MAX_SAMPLES = 600
SHAP_TOP_N = 25
DEPENDENCE_TOP_N = 5

OUTDIR = "analysis_outputs/11_shap_beeswarm"


def main():
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Install with: pip install shap") from e

    outdir = ensure_dir(OUTDIR)

    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    idx_all = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = make_rf(random_state=RANDOM_STATE, n_estimators=RF_N_ESTIMATORS)
    model.fit(X_train, y_train)
    joblib.dump(model, outdir / "rf_model_used_for_shap.joblib")

    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)

    metrics = {
        "ROC": roc_auc_score(y_test, y_score),
        "AP": average_precision_score(y_test, y_score),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(outdir / "holdout_metrics_for_shap_model.csv", index=False)

    pred_df = pd.DataFrame({
        "OriginalIndex": idx_test,
        "SMILES": df.iloc[idx_test][smiles_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score_class1": y_score,
    })
    pred_df.to_csv(outdir / "test_predictions_for_shap.csv", index=False)

    rng = np.random.RandomState(RANDOM_STATE)
    if X_test.shape[0] > SHAP_MAX_SAMPLES:
        local_idx = rng.choice(X_test.shape[0], size=SHAP_MAX_SAMPLES, replace=False)
    else:
        local_idx = np.arange(X_test.shape[0])
    X_shap = X_test[local_idx]

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_shap)
    shap_values, base_value = extract_class1_shap(raw_shap, explainer.expected_value)

    np.save(outdir / "shap_values_class1.npy", shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": feature_names, "mean_abs_SHAP": mean_abs})
    shap_df = shap_df.sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    shap_df.to_csv(outdir / "shap_top_features.csv", index=False)

    plt.figure(figsize=(10, 9))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        max_display=SHAP_TOP_N,
        show=False,
    )
    plt.title(f"SHAP Beeswarm — RF classifier, class 1 [top {SHAP_TOP_N}]", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_beeswarm_top{SHAP_TOP_N}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 9))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        max_display=SHAP_TOP_N,
        show=False,
    )
    plt.title(f"SHAP Mean |Value| — RF classifier, class 1 [top {SHAP_TOP_N}]", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_bar_top{SHAP_TOP_N}.png", dpi=300, bbox_inches="tight")
    plt.close()

    for feat in shap_df["Feature"].head(DEPENDENCE_TOP_N):
        fi = feature_names.index(feat)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            fi,
            shap_values,
            X_shap,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP dependence — {feat}")
        plt.tight_layout()
        plt.savefig(outdir / f"shap_dependence_{safe_filename(feat)}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(shap_df.head(SHAP_TOP_N).to_string(index=False))
    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
