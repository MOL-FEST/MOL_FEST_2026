#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_lime_local_explanations.py
=============================

Trains a Random Forest classifier and creates LIME local explanations.

Outputs:
  analysis_outputs/13_lime/
    lime_sample_*.png
    lime_sample_*.html
    lime_all_local_explanations.csv
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
    load_features_from_csv, make_rf, positive_scores,
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

LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 2000
MAX_CASES = 5

OUTDIR = "analysis_outputs/13_lime"


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
    add("false_positive", (y_true == 0) & (y_pred == 1), y_score, True)
    add("false_negative", (y_true == 1) & (y_pred == 0), y_score, False)
    add("borderline_near_0p50", np.ones_like(y_true, dtype=bool), np.abs(y_score - 0.5), False)

    out = []
    seen = set()
    for label, i in cases:
        if i not in seen:
            out.append((label, i))
            seen.add(i)
    return out[:MAX_CASES]


def main():
    try:
        import lime
        import lime.lime_tabular
    except ImportError as e:
        raise ImportError("LIME is not installed. Install with: pip install lime") from e

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
    joblib.dump(model, outdir / "rf_model_used_for_lime.joblib")

    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(feature_names),
        class_names=["class_0", "class_1"],
        discretize_continuous=False,
        mode="classification",
        random_state=RANDOM_STATE,
    )

    rows = []
    for case_no, (case_label, local_i) in enumerate(choose_cases(y_test, y_pred, y_score), start=1):
        original_idx = int(idx_test[local_i])
        exp = explainer.explain_instance(
            data_row=X_test[local_i],
            predict_fn=model.predict_proba,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
            labels=(1,),
        )

        html_name = f"lime_sample_{case_no}_{safe_filename(case_label)}_idx_{original_idx}.html"
        png_name = f"lime_sample_{case_no}_{safe_filename(case_label)}_idx_{original_idx}.png"
        exp.save_to_file(str(outdir / html_name))

        weights = sorted(exp.as_list(label=1), key=lambda z: abs(z[1]), reverse=True)
        labels = [z[0] for z in weights]
        vals = [z[1] for z in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = np.arange(len(labels))
        ax.barh(pos, vals)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("LIME weight for class 1")
        ax.set_title(
            f"LIME — {case_label}\n"
            f"idx={original_idx}, true={y_test[local_i]}, pred={y_pred[local_i]}, score={y_score[local_i]:.3f}"
        )
        plt.tight_layout()
        plt.savefig(outdir / png_name, dpi=300, bbox_inches="tight")
        plt.close()

        for rank, (feature_rule, weight) in enumerate(weights, start=1):
            rows.append({
                "case_no": case_no,
                "case": case_label,
                "original_index": original_idx,
                "y_true": int(y_test[local_i]),
                "y_pred": int(y_pred[local_i]),
                "y_score_class1": float(y_score[local_i]),
                "rank": rank,
                "feature_rule": feature_rule,
                "lime_weight_class1": float(weight),
                "png_file": png_name,
                "html_file": html_name,
            })

    pd.DataFrame(rows).to_csv(outdir / "lime_all_local_explanations.csv", index=False)
    print(f"[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
