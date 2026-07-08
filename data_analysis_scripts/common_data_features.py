#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_common_data_features.py
==========================

Common helpers used by the individual visual-analysis scripts.

The scripts support two input styles:
1) CSV has SMILES only  -> features are generated from SMILES using RDKit.
2) CSV already has fingerprint columns such as MACCS_, Avalon_FP_, Avalon_b,
   morgan_ -> those columns can be used directly when FEATURE_MODE="auto".

Default target/smiles columns match the ERα agonist/antagonist file used in
the previous workshop scripts.
"""

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FINGERPRINT_PREFIXES = ("MACCS_", "Avalon_FP_", "Avalon_b", "morgan_")


def resolve_input_file(path_like):
    p = Path(path_like)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([Path.cwd() / p, Path(__file__).resolve().parent / p, Path("/mnt/data") / p])
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Input file not found. Tried: " + ", ".join(str(c) for c in candidates))


def read_table(path, sep):
    path = Path(path)
    if sep is None:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)


def detect_target_column(df):
    exact = [
        "binary_label_agonist1_antagonist0",
        "binary_label_active1_inactive0",
        "active",
        "Active",
        "label",
        "Label",
        "target",
        "Target",
        "class",
        "Class",
        "y",
        "Y",
    ]
    for c in exact:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "binary_label" in cl or "active1" in cl or "label" in cl or "target" in cl:
            return c
    raise ValueError("Target column could not be detected. Set TARGET_COLUMN manually.")


def detect_smiles_column(df):
    exact = [
        "QSAR-Ready SMILES",
        "SMILES",
        "smiles",
        "canonical_smiles",
        "canonical_smiles_clean",
        "standardized_smiles",
    ]
    for c in exact:
        if c in df.columns:
            return c
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    raise ValueError("SMILES column could not be detected. Set SMILES_COLUMN manually.")


def prepare_binary_target(df, target_col, positive_threshold=None):
    if target_col not in df.columns:
        raise ValueError(f"Target column was not found: {target_col}")

    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    keep = y_raw.notna()
    df2 = df.loc[keep].copy()
    y_raw = y_raw.loc[keep]

    if positive_threshold is not None:
        y = (y_raw.astype(float).to_numpy() >= float(positive_threshold)).astype(int)
        return df2, y

    classes = np.sort(y_raw.unique())
    if len(classes) != 2:
        raise ValueError(
            "This script expects a binary target. "
            f"Found {len(classes)} unique values. "
            "If your target is continuous pActivity, set POSITIVE_THRESHOLD, e.g. 6.0."
        )

    if set(classes).issubset({0, 1}):
        y = y_raw.astype(int).to_numpy()
    else:
        mapping = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(mapping).astype(int).to_numpy()
    return df2, y


def existing_fingerprint_columns(df, target_col=None, smiles_col=None):
    exclude = {target_col, smiles_col, None, "class_code", "class_label", "activity_comment", "standard_relation"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(str(c).startswith(prefix) for prefix in FINGERPRINT_PREFIXES):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
    return sorted(cols)


def smiles_to_fingerprints(smiles, fingerprint_type="maccs", morgan_bits=1024, morgan_radius=2):
    fingerprint_type = fingerprint_type.lower().strip()
    if fingerprint_type not in {"maccs", "morgan"}:
        raise ValueError("fingerprint_type must be 'maccs' or 'morgan'.")

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    except ImportError as e:
        raise ImportError("RDKit is required. Install with: pip install rdkit") from e

    if fingerprint_type == "maccs":
        feature_names = [f"MACCS_{i}" for i in range(1, 167)]
        generator = None
    else:
        feature_names = [f"morgan_{i}" for i in range(morgan_bits)]
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)

    rows = []
    valid_mask = []
    for i, smi in enumerate(smiles, start=1):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            continue

        if fingerprint_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr167 = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr167)
            arr = arr167[1:]  # bit 0 is not a real MACCS key
        else:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((morgan_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)

        rows.append(arr)
        valid_mask.append(True)

        if i % 1000 == 0:
            print(f"Generated fingerprint features for {i}/{len(smiles)} molecules...")

    if len(rows) == 0:
        raise ValueError("No valid SMILES could be parsed.")

    return np.vstack(rows).astype(np.float32), feature_names, np.array(valid_mask, dtype=bool)


def load_features_from_csv(
    input_file,
    csv_separator,
    target_column,
    smiles_column,
    feature_mode,
    fingerprint_type,
    morgan_bits,
    morgan_radius,
    positive_threshold,
    use_feature_cache=True,
):
    input_path = resolve_input_file(input_file)
    df = read_table(input_path, csv_separator)

    if target_column in [None, "AUTO"]:
        target_column = detect_target_column(df)
    if smiles_column in [None, "AUTO"]:
        smiles_column = detect_smiles_column(df)

    df, y = prepare_binary_target(df, target_column, positive_threshold=positive_threshold)

    feature_mode = feature_mode.lower().strip()
    if feature_mode not in {"auto", "existing", "smiles"}:
        raise ValueError("FEATURE_MODE must be 'auto', 'existing', or 'smiles'.")

    fp_cols = existing_fingerprint_columns(df, target_col=target_column, smiles_col=smiles_column)

    if feature_mode == "existing" or (feature_mode == "auto" and len(fp_cols) > 0):
        X_df = df[fp_cols].apply(pd.to_numeric, errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

        # Drop constant columns for numerical stability in correlation/projection.
        std = X_df.std(axis=0)
        keep = std > 0
        X_df = X_df.loc[:, keep]

        X = X_df.values.astype(np.float32)
        feature_names = list(X_df.columns)
        df = df.reset_index(drop=True)
        return df, X, y, feature_names, target_column, smiles_column, input_path

    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column was not found: {smiles_column}")

    cache_path = input_path.with_suffix(f".{fingerprint_type}_features_m{morgan_bits}_r{morgan_radius}.npz")
    if use_feature_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        feature_names = cached["feature_names"].tolist()
        valid_mask = cached["valid_mask"]
    else:
        X, feature_names, valid_mask = smiles_to_fingerprints(
            df[smiles_column].tolist(),
            fingerprint_type=fingerprint_type,
            morgan_bits=morgan_bits,
            morgan_radius=morgan_radius,
        )
        if use_feature_cache:
            np.savez_compressed(
                cache_path,
                X=X,
                feature_names=np.array(feature_names, dtype=object),
                valid_mask=valid_mask,
            )

    df = df.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    return df, X, y, feature_names, target_column, smiles_column, input_path


def make_rf(random_state=42, n_jobs=-1, n_estimators=500):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return model.predict(X).astype(float)


def extract_class1_shap(shap_values, expected_value):
    arr = shap_values
    if isinstance(arr, list):
        values = arr[1] if len(arr) > 1 else arr[0]
    else:
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            values = arr[:, :, 1]
        elif arr.ndim == 3 and arr.shape[0] == 2:
            values = arr[1, :, :]
        else:
            values = arr

    ev = np.asarray(expected_value)
    if ev.ndim == 0:
        base = float(ev)
    elif ev.size > 1:
        base = float(ev.ravel()[1])
    else:
        base = float(ev.ravel()[0])

    return np.asarray(values, dtype=float), base


def safe_filename(text, max_len=120):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len]


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def top_anova_features(X, y, feature_names, top_n):
    from sklearn.feature_selection import f_classif
    F, p = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    out = pd.DataFrame({"Feature": feature_names, "F_score": F, "p_value": p})
    out = out.sort_values("F_score", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out, out.head(min(top_n, len(out)))["Feature"].tolist()
