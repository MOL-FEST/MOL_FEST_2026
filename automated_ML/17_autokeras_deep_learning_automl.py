#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_autokeras_deep_learning_automl.py
====================================

AutoKeras automated deep-learning classification for molecular fingerprints.

Çıktılar:
  advanced_outputs/17_autokeras/
    autokeras_tuner_project/
    autokeras_best_model.keras
    autokeras_test_predictions.csv
    autokeras_test_metrics.csv
    metadata.json
    autokeras_feature_metadata.joblib

Kurulum:
  pip install autokeras tensorflow pandas numpy scikit-learn rdkit joblib
"""

from __future__ import annotations

import json
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from common_data_features import load_features_from_csv, ensure_dir

# =========================
# USER SETTINGS
# =========================
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"
TARGET_COLUMN = "binary_label_agonist1_antagonist0"  # or "AUTO"
SMILES_COLUMN = "QSAR-Ready SMILES"                   # or "AUTO"

FEATURE_MODE = "smiles"       # "smiles", "existing", or "auto"
FINGERPRINT_TYPE = "maccs"    # "maccs" or "morgan"
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
POSITIVE_THRESHOLD = None

TEST_SIZE = 0.20
RANDOM_STATE = 42
OUTDIR = "advanced_outputs/17_autokeras"
PROJECT_NAME = "autokeras_structured_classifier"

MAX_TRIALS = 20
EPOCHS = 50
VALIDATION_SPLIT = 0.20
OVERWRITE_PROJECT = True
USE_CLASS_WEIGHT = True


def safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return np.nan


def convert_autokeras_predictions_to_labels(raw_pred):
    arr = np.asarray(raw_pred).ravel()
    s = pd.to_numeric(pd.Series(arr), errors="coerce")
    if s.notna().all():
        vals = s.astype(float).to_numpy()
        if np.any((vals > 0) & (vals < 1)):
            return (vals >= 0.5).astype(int)
        return vals.astype(int)
    return pd.Series(arr).astype(str).map({"0": 0, "1": 1, "class_0": 0, "class_1": 1}).fillna(0).astype(int).to_numpy()


def get_score_from_autokeras(clf, exported_model, X_test_df, y_pred):
    try:
        proba = clf.predict_proba(X_test_df)
        arr = np.asarray(proba)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(float)
    except Exception:
        pass

    for x_in in [X_test_df, X_test_df.values.astype("float32")]:
        try:
            raw = exported_model.predict(x_in, verbose=0)
            arr = np.asarray(raw)
            if arr.ndim == 1:
                score = arr.astype(float)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                score = arr[:, 0].astype(float)
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                score = arr[:, 1].astype(float)
            else:
                continue
            if np.nanmin(score) < 0 or np.nanmax(score) > 1:
                score = 1.0 / (1.0 + np.exp(-score))
            return score
        except Exception:
            continue

    return np.asarray(y_pred, dtype=float)


def main():
    try:
        import tensorflow as tf
        import autokeras as ak
    except ImportError as e:
        raise ImportError("AutoKeras/TensorFlow is not installed. Install with: pip install autokeras tensorflow") from e

    try:
        tf.keras.utils.set_random_seed(RANDOM_STATE)
    except Exception:
        pass

    outdir = ensure_dir(OUTDIR)
    tuner_dir = ensure_dir(outdir / "autokeras_tuner_project")

    print("[1] Loading data and features...")
    df, X, y, feature_names, target_col, smiles_col, input_path = load_features_from_csv(
        INPUT_FILE, CSV_SEPARATOR, TARGET_COLUMN, SMILES_COLUMN,
        FEATURE_MODE, FINGERPRINT_TYPE, MORGAN_BITS, MORGAN_RADIUS,
        POSITIVE_THRESHOLD,
    )

    idx_all = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    metadata = {
        "input_file": str(input_path),
        "target_column": target_col,
        "smiles_column": smiles_col,
        "feature_mode": FEATURE_MODE,
        "fingerprint_type": FINGERPRINT_TYPE,
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "n_rows_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "max_trials": MAX_TRIALS,
        "epochs": EPOCHS,
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump({"feature_names": feature_names, "metadata": metadata}, outdir / "autokeras_feature_metadata.joblib")

    print(f"Rows: {len(y)} | Train: {len(y_train)} | Test: {len(y_test)} | Features: {X.shape[1]}")
    print("Class counts:")
    print(pd.Series(y).value_counts().sort_index())

    class_weight = None
    if USE_CLASS_WEIGHT:
        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"Class weights: {class_weight}")

    print("\n[2] Running AutoKeras StructuredDataClassifier...")
    clf = ak.StructuredDataClassifier(
        max_trials=MAX_TRIALS,
        overwrite=OVERWRITE_PROJECT,
        directory=str(tuner_dir),
        project_name=PROJECT_NAME,
        seed=RANDOM_STATE,
    )

    fit_kwargs = {}
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    clf.fit(
        x=X_train_df,
        y=y_train,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        **fit_kwargs,
    )

    print("\n[3] Exporting best Keras model...")
    exported_model = clf.export_model()
    saved_model_path = outdir / "autokeras_best_model.keras"
    try:
        exported_model.save(saved_model_path)
        print(f"Saved exported model: {saved_model_path}")
    except Exception as e:
        print(f"Saving .keras failed: {e}")
        saved_model_dir = outdir / "autokeras_best_model_savedmodel"
        exported_model.save(saved_model_dir)
        print(f"Saved exported model directory: {saved_model_dir}")

    print("\n[4] Test predictions...")
    raw_pred = clf.predict(X_test_df)
    y_pred = convert_autokeras_predictions_to_labels(raw_pred)
    y_score = get_score_from_autokeras(clf, exported_model, X_test_df, y_pred)

    pred_df = pd.DataFrame({
        "OriginalIndex": idx_test,
        "SMILES": df.iloc[idx_test][smiles_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score_class1": y_score,
    })
    pred_df.to_csv(outdir / "autokeras_test_predictions.csv", index=False)

    metrics = {
        "ROC": safe_metric(roc_auc_score, y_test, y_score),
        "AP": safe_metric(average_precision_score, y_test, y_score),
        "F1": safe_metric(f1_score, y_test, y_pred),
        "Accuracy": safe_metric(accuracy_score, y_test, y_pred),
        "Recall": safe_metric(recall_score, y_test, y_pred),
        "Precision": safe_metric(precision_score, y_test, y_pred),
    }
    pd.DataFrame([metrics]).to_csv(outdir / "autokeras_test_metrics.csv", index=False)

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n[DONE]")
    print(f"Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
