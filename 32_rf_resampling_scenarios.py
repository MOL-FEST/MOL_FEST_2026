#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_rf_resampling_scenarios.py
=============================

Random Forest classification with training-set resampling scenarios.

Purpose
-------
This script reads the original molecular CSV, generates fingerprint features
from SMILES, splits into train/test, applies resampling ONLY on the training set,
trains Random Forest models, and saves models + predictions + metrics.

Scenarios
---------
For each target class ratio, the script runs BOTH oversampling and undersampling:

1) balanced_1_to_1
   positive : negative ≈ 1 : 1

2) positive_5_to_1
   positive : negative ≈ 5 : 1

3) negative_5_to_1
   positive : negative ≈ 1 : 5

Why both over/under?
--------------------
- Oversampling duplicates minority/needed-class samples until the requested ratio
  is reached. It does not remove training samples.
- Undersampling removes samples from the overrepresented class until the requested
  ratio is reached. It does not duplicate samples.

Important anti-leakage rule
---------------------------
Resampling is applied AFTER train/test split and ONLY to the training set.
The test set is never resampled.

Outputs
-------
resampling_outputs/32_rf_resampling/
  rf_resampling_metrics.csv
  all_predictions_long.csv
  train_resampled_class_counts.csv
  train_test_indices.csv
  feature_names.csv
  saved_models/*.joblib
  predictions/*.csv
  feature_importances/*.csv

Install
-------
pip install pandas numpy scikit-learn rdkit joblib

Run
---
python 32_rf_resampling_scenarios.py
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"

TARGET_COLUMN = "binary_label_agonist1_antagonist0"  # or "AUTO"
SMILES_COLUMN = "QSAR-Ready SMILES"                   # or "AUTO"

# If target is already 0/1, keep None.
# If target is pActivity and you want active/inactive labels, set e.g. 6.0.
POSITIVE_THRESHOLD = None

# Feature settings
FINGERPRINT_TYPE = "morgan"  # "morgan" or "maccs"
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
USE_FEATURE_CACHE = True

# Split/model settings
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 500
N_JOBS = -1
USE_CLASS_WEIGHT = False  # keep False to isolate the effect of resampling

# For undersampling, very extreme ratios can leave too few samples in one class.
# This minimum prevents impossible/toy training sets.
MIN_CLASS_COUNT_AFTER_UNDERSAMPLING = 5

OUTDIR = "resampling_outputs/32_rf_resampling"


# =============================================================================
# Basic IO / detection
# =============================================================================
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(text: str, max_len: int = 140) -> str:
    import re
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len]


def resolve_input_file(path_like: str | Path) -> Path:
    p = Path(path_like)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([Path.cwd() / p, Path(__file__).resolve().parent / p, Path("/mnt/data") / p])
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Input file not found. Tried: " + ", ".join(str(c) for c in candidates))


def read_table(path: str | Path, sep: str | None) -> pd.DataFrame:
    if sep is None:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)


def detect_target_column(df: pd.DataFrame) -> str:
    exact = [
        "binary_label_agonist1_antagonist0",
        "binary_label_active1_inactive0",
        "active", "Active",
        "label", "Label",
        "target", "Target",
        "class", "Class",
        "y", "Y",
    ]
    for c in exact:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "binary_label" in cl or "active1" in cl or "label" in cl or "target" in cl:
            return c
    raise ValueError("Target column could not be detected. Set TARGET_COLUMN manually.")


def detect_smiles_column(df: pd.DataFrame) -> str:
    exact = [
        "QSAR-Ready SMILES",
        "SMILES",
        "smiles",
        "canonical_smiles",
        "standardized_smiles",
        "canonical_smiles_clean",
    ]
    for c in exact:
        if c in df.columns:
            return c
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    raise ValueError("SMILES column could not be detected. Set SMILES_COLUMN manually.")


def prepare_binary_target(df: pd.DataFrame, target_col: str, positive_threshold: float | None):
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    keep = y_raw.notna()
    df2 = df.loc[keep].copy().reset_index(drop=True)
    y_raw = y_raw.loc[keep].reset_index(drop=True)

    if positive_threshold is not None:
        y = (y_raw.astype(float).to_numpy() >= float(positive_threshold)).astype(int)
        return df2, y

    classes = np.sort(y_raw.dropna().unique())
    if len(classes) != 2:
        raise ValueError(
            f"Expected binary target with exactly two classes, found {len(classes)} classes. "
            "If this is continuous pActivity, set POSITIVE_THRESHOLD = 6.0 or another threshold."
        )

    if set(classes).issubset({0, 1}):
        y = y_raw.astype(int).to_numpy()
    else:
        mapping = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(mapping).astype(int).to_numpy()

    return df2, y


# =============================================================================
# Fingerprint features
# =============================================================================
def smiles_to_features(
    smiles: list[str],
    fingerprint_type: str,
    morgan_bits: int,
    morgan_radius: int,
    cache_path: Path | None = None,
    use_cache: bool = True,
):
    fingerprint_type = fingerprint_type.lower().strip()
    if fingerprint_type not in {"morgan", "maccs"}:
        raise ValueError("FINGERPRINT_TYPE must be 'morgan' or 'maccs'.")

    if use_cache and cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["feature_names"].tolist(), cached["valid_mask"]

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    except ImportError as e:
        raise ImportError("RDKit is required. Install with: pip install rdkit") from e

    if fingerprint_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
        feature_names = [f"Morgan_r{morgan_radius}_{i}" for i in range(morgan_bits)]
    else:
        generator = None
        feature_names = [f"MACCS_{i}" for i in range(1, 167)]

    rows = []
    valid_mask = []

    for i, smi in enumerate(smiles, start=1):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            continue

        if fingerprint_type == "morgan":
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((morgan_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr167 = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr167)
            arr = arr167[1:]  # remove non-informative bit 0

        rows.append(arr)
        valid_mask.append(True)

        if i % 1000 == 0:
            print(f"Generated features for {i}/{len(smiles)} molecules...")

    if not rows:
        raise ValueError("No valid SMILES could be parsed.")

    X = np.vstack(rows).astype(np.float32)
    valid_mask = np.array(valid_mask, dtype=bool)

    if use_cache and cache_path is not None:
        np.savez_compressed(
            cache_path,
            X=X,
            feature_names=np.array(feature_names, dtype=object),
            valid_mask=valid_mask,
        )

    return X, feature_names, valid_mask


def load_dataset():
    input_path = resolve_input_file(INPUT_FILE)
    df = read_table(input_path, CSV_SEPARATOR)

    target_col = detect_target_column(df) if TARGET_COLUMN in [None, "AUTO"] else TARGET_COLUMN
    smiles_col = detect_smiles_column(df) if SMILES_COLUMN in [None, "AUTO"] else SMILES_COLUMN

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")

    df, y = prepare_binary_target(df, target_col, POSITIVE_THRESHOLD)

    cache_path = input_path.with_suffix(
        f".{FINGERPRINT_TYPE}_m{MORGAN_BITS}_r{MORGAN_RADIUS}_resampling_features.npz"
    )
    X, feature_names, valid_mask = smiles_to_features(
        df[smiles_col].tolist(),
        fingerprint_type=FINGERPRINT_TYPE,
        morgan_bits=MORGAN_BITS,
        morgan_radius=MORGAN_RADIUS,
        cache_path=cache_path,
        use_cache=USE_FEATURE_CACHE,
    )

    df = df.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    return df, X, y, feature_names, target_col, smiles_col, input_path


# =============================================================================
# Resampling
# =============================================================================
def class_counts(y):
    y = np.asarray(y)
    return {
        0: int(np.sum(y == 0)),
        1: int(np.sum(y == 1)),
    }


def resample_to_ratio(
    y_train: np.ndarray,
    target_pos_to_neg_ratio: float,
    method: str,
    random_state: int = 42,
    min_class_count_after_undersampling: int = 5,
):
    """Return resampled indices into the original training array.

    Parameters
    ----------
    target_pos_to_neg_ratio:
        Desired positive:negative ratio.
        1.0 = balanced
        5.0 = positives five times negatives
        0.2 = negatives five times positives
    method:
        "oversampling" or "undersampling"
    """
    method = method.lower().strip()
    if method not in {"oversampling", "undersampling"}:
        raise ValueError("method must be 'oversampling' or 'undersampling'.")

    rng = np.random.RandomState(random_state)

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Both classes are required. Found positives={n_pos}, negatives={n_neg}")

    r = float(target_pos_to_neg_ratio)
    if r <= 0:
        raise ValueError("target_pos_to_neg_ratio must be positive.")

    if method == "oversampling":
        # Only add samples; do not remove any original training sample.
        # If current ratio is too low, duplicate positives. If too high, duplicate negatives.
        current_ratio = n_pos / n_neg

        if current_ratio < r:
            target_pos = int(np.ceil(r * n_neg))
            target_neg = n_neg
        else:
            target_pos = n_pos
            target_neg = int(np.ceil(n_pos / r))

        sampled_pos = rng.choice(pos_idx, size=target_pos, replace=(target_pos > n_pos))
        sampled_neg = rng.choice(neg_idx, size=target_neg, replace=(target_neg > n_neg))

    else:
        # Only remove samples; do not duplicate any training sample.
        current_ratio = n_pos / n_neg

        if current_ratio < r:
            # Need fewer negatives relative to positives.
            target_pos = n_pos
            target_neg = int(np.floor(n_pos / r))
            target_neg = max(min_class_count_after_undersampling, target_neg)
            target_neg = min(target_neg, n_neg)
        else:
            # Need fewer positives relative to negatives.
            target_neg = n_neg
            target_pos = int(np.floor(r * n_neg))
            target_pos = max(min_class_count_after_undersampling, target_pos)
            target_pos = min(target_pos, n_pos)

        sampled_pos = rng.choice(pos_idx, size=target_pos, replace=False)
        sampled_neg = rng.choice(neg_idx, size=target_neg, replace=False)

    sampled = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled)
    return sampled


# =============================================================================
# Evaluation
# =============================================================================
def make_rf():
    if USE_CLASS_WEIGHT:
        class_weight = "balanced_subsample"
    else:
        class_weight = None

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return model.predict(X).astype(float)


def safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return np.nan


def metric_row(
    scenario_name,
    resampling_method,
    target_ratio,
    y_true,
    y_pred,
    y_score,
    train_counts_before,
    train_counts_after,
):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "Scenario": scenario_name,
        "ResamplingMethod": resampling_method,
        "TargetPosNegRatio": target_ratio,
        "TrainBefore_Positive": train_counts_before[1],
        "TrainBefore_Negative": train_counts_before[0],
        "TrainAfter_Positive": train_counts_after[1],
        "TrainAfter_Negative": train_counts_after[0],
        "TrainAfter_PosNegRatio": train_counts_after[1] / train_counts_after[0] if train_counts_after[0] else np.nan,
        "ROC": safe_metric(roc_auc_score, y_true, y_score),
        "AP": safe_metric(average_precision_score, y_true, y_score),
        "F1": safe_metric(f1_score, y_true, y_pred),
        "Accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "BalancedAccuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "RecallSensitivity": safe_metric(recall_score, y_true, y_pred),
        "Specificity": specificity,
        "Precision": safe_metric(precision_score, y_true, y_pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    outdir = ensure_dir(OUTDIR)
    models_dir = ensure_dir(outdir / "saved_models")
    pred_dir = ensure_dir(outdir / "predictions")
    fi_dir = ensure_dir(outdir / "feature_importances")

    print("[1] Loading data and generating features...")
    df, X, y, feature_names, target_col, smiles_col, input_path = load_dataset()

    print(f"Input: {input_path}")
    print(f"Rows after target/SMILES filtering: {len(y)}")
    print(f"Features: {X.shape[1]} ({FINGERPRINT_TYPE})")
    print(f"Target: {target_col}")
    print(f"SMILES: {smiles_col}")
    print("Overall class counts:")
    print(pd.Series(y).value_counts().sort_index().rename({0: "negative_0", 1: "positive_1"}))

    all_idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        all_idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    split_df = pd.DataFrame({
        "OriginalIndex": np.concatenate([idx_train, idx_test]),
        "Split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
    })
    split_df.to_csv(outdir / "train_test_indices.csv", index=False)

    pd.DataFrame({"Feature": feature_names}).to_csv(outdir / "feature_names.csv", index=False)

    metadata = {
        "input_file": str(input_path),
        "target_column": target_col,
        "smiles_column": smiles_col,
        "fingerprint_type": FINGERPRINT_TYPE,
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "n_rows": int(len(y)),
        "n_train_before_resampling": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "rf_n_estimators": N_ESTIMATORS,
        "use_class_weight": USE_CLASS_WEIGHT,
        "note": "Resampling is applied only to the training split. The test split is not resampled.",
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    scenarios = [
        ("balanced_1_to_1", 1.0),
        ("positive_5_to_1", 5.0),
        ("negative_5_to_1", 1.0 / 5.0),
    ]
    methods = ["oversampling", "undersampling"]

    train_counts_before = class_counts(y_train)

    metrics_rows = []
    all_predictions = []
    count_rows = []

    print("\n[2] Running RF models for resampling scenarios...")

    for scenario_name, target_ratio in scenarios:
        for method in methods:
            run_name = f"{scenario_name}_{method}"
            print("\n" + "=" * 80)
            print(f"Scenario: {scenario_name} | Method: {method} | Target pos:neg = {target_ratio:g}")
            print("=" * 80)

            sampled_train_local_idx = resample_to_ratio(
                y_train,
                target_pos_to_neg_ratio=target_ratio,
                method=method,
                random_state=RANDOM_STATE,
                min_class_count_after_undersampling=MIN_CLASS_COUNT_AFTER_UNDERSAMPLING,
            )

            X_res = X_train[sampled_train_local_idx]
            y_res = y_train[sampled_train_local_idx]
            original_train_indices_resampled = idx_train[sampled_train_local_idx]

            train_counts_after = class_counts(y_res)

            count_rows.append({
                "Scenario": scenario_name,
                "ResamplingMethod": method,
                "TargetPosNegRatio": target_ratio,
                "Before_Positive": train_counts_before[1],
                "Before_Negative": train_counts_before[0],
                "After_Positive": train_counts_after[1],
                "After_Negative": train_counts_after[0],
                "After_Total": int(len(y_res)),
                "After_PosNegRatio": train_counts_after[1] / train_counts_after[0] if train_counts_after[0] else np.nan,
            })

            print("Training counts before resampling:", train_counts_before)
            print("Training counts after  resampling:", train_counts_after)

            # Save resampled training index list. Oversampling can contain duplicated OriginalIndex values.
            resampled_index_df = pd.DataFrame({
                "Scenario": scenario_name,
                "ResamplingMethod": method,
                "ResampledTrainRowOrder": np.arange(len(original_train_indices_resampled)),
                "OriginalIndex": original_train_indices_resampled,
                "y": y_res,
            })
            resampled_index_df.to_csv(outdir / f"resampled_train_indices_{run_name}.csv", index=False)

            model = make_rf()
            model.fit(X_res, y_res)

            y_pred = model.predict(X_test)
            y_score = positive_scores(model, X_test)

            row = metric_row(
                scenario_name=scenario_name,
                resampling_method=method,
                target_ratio=target_ratio,
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                train_counts_before=train_counts_before,
                train_counts_after=train_counts_after,
            )
            metrics_rows.append(row)

            print("Test metrics:")
            for k in ["ROC", "AP", "F1", "Accuracy", "BalancedAccuracy", "RecallSensitivity", "Specificity", "Precision"]:
                print(f"  {k}: {row[k]:.4f}")

            safe_run_name = safe_filename(run_name)
            joblib.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "metadata": metadata,
                    "scenario": scenario_name,
                    "resampling_method": method,
                    "target_pos_neg_ratio": target_ratio,
                    "train_counts_before": train_counts_before,
                    "train_counts_after": train_counts_after,
                },
                models_dir / f"{safe_run_name}_rf.joblib",
            )

            pred_df = pd.DataFrame({
                "Scenario": scenario_name,
                "ResamplingMethod": method,
                "TargetPosNegRatio": target_ratio,
                "OriginalIndex": idx_test,
                "SMILES": df.iloc[idx_test][smiles_col].values,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_score_class1": y_score,
            })
            pred_df.to_csv(pred_dir / f"predictions_{safe_run_name}.csv", index=False)
            all_predictions.append(pred_df)

            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "Feature": feature_names,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                fi["Rank"] = np.arange(1, len(fi) + 1)
                fi.to_csv(fi_dir / f"feature_importance_{safe_run_name}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["ROC", "AP"], ascending=False)
    metrics_df.to_csv(outdir / "rf_resampling_metrics.csv", index=False)

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(outdir / "train_resampled_class_counts.csv", index=False)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(outdir / "all_predictions_long.csv", index=False)

    display_cols = [
        "Scenario",
        "ResamplingMethod",
        "TrainAfter_Positive",
        "TrainAfter_Negative",
        "TrainAfter_PosNegRatio",
        "ROC",
        "AP",
        "F1",
        "Accuracy",
        "BalancedAccuracy",
        "RecallSensitivity",
        "Specificity",
        "Precision",
    ]

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(metrics_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  Metrics:      {outdir / 'rf_resampling_metrics.csv'}")
    print(f"  Predictions:  {outdir / 'all_predictions_long.csv'}")
    print(f"  Models:       {models_dir}")
    print(f"  Importances:  {fi_dir}")
    print(f"\n[DONE] Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
