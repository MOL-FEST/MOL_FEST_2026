#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_shap_waterfall_cases.py
==========================

Trains a Random Forest classifier and creates SHAP waterfall plots for selected cases:
- correct high-confidence positive
- correct high-confidence negative
- false positive, if present
- false negative, if present
- borderline prediction near 0.50

Output folder:
  analysis_outputs/12_shap_waterfall/
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
MAX_DISPLAY_FEATURES = 15

OUTDIR = "analysis_outputs/12_shap_waterfall"


def choose_cases(y_true, y_pred, y_score):
    cases = []

    def add(label, mask, values, reverse=True):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        order = np.argsort(values[idx])
        if reverse:
            order = order[::-1]
        cases.append((label, int(idx[order[0]])))

    add("correct_high_confidence_positive", (y_true == 1) & (y_pred == 1), y_score, True)
    add("correct_high_confidence_negative", (y_true == 0) & (y_pred == 0), 1 - y_score, True)
    add("false_positive_high_score", (y_true == 0) & (y_pred == 1), y_score, True)
    add("false_negative_low_score", (y_true == 1) & (y_pred == 0), y_score, False)
    add("borderline_prediction_near_0p50", np.ones_like(y_true, dtype=bool), np.abs(y_score - 0.5), False)

    out = []
    seen = set()
    for label, i in cases:
        if i not in seen:
            out.append((label, i))
            seen.add(i)
    return out


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
    joblib.dump(model, outdir / "rf_model_used_for_waterfall.joblib")

    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test)
    shap_values, base_value = extract_class1_shap(raw_shap, explainer.expected_value)

    exp = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(base_value, X_test.shape[0]),
        data=X_test,
        feature_names=feature_names,
    )

    rows = []
    for case_label, local_i in choose_cases(y_test, y_pred, y_score):
        original_idx = int(idx_test[local_i])
        file_name = f"shap_waterfall_{safe_filename(case_label)}_idx_{original_idx}.png"

        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(exp[local_i], max_display=MAX_DISPLAY_FEATURES, show=False)
        plt.title(
            f"SHAP Waterfall — {case_label}\n"
            f"idx={original_idx}, true={y_test[local_i]}, pred={y_pred[local_i]}, score={y_score[local_i]:.3f}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(outdir / file_name, dpi=300, bbox_inches="tight")
        plt.close()

        rows.append({
            "case": case_label,
            "original_index": original_idx,
            "y_true": int(y_test[local_i]),
            "y_pred": int(y_pred[local_i]),
            "y_score_class1": float(y_score[local_i]),
            "file": file_name,
        })

    pd.DataFrame(rows).to_csv(outdir / "shap_waterfall_cases.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
