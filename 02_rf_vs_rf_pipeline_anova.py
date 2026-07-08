"""
02_rf_vs_rf_pipeline_anova.py

Purpose
-------
A focused Random Forest demonstration for classification:

1) Bare RF only: RandomForestClassifier directly on all SMILES-derived features.
2) RF pipeline: StandardScaler -> ANOVA SelectKBest -> RandomForestClassifier
   with k = 50, 100, 150 selected features.

The key teaching point is that Random Forest can already give strong/promising
performance without StandardScaler and without ANOVA feature selection. The
pipeline variants are included to test whether aggressive univariate feature
selection improves, matches, or harms performance.

Metrics: ROC-AUC, Average Precision, F1, Accuracy, Recall, Precision.
Evaluation: Stratified 5-fold CV with mean +/- SD.

How to run
----------
1) Put this .py file and your CSV in the same folder, or edit INPUT_FILE below.
2) Install dependencies if needed:
   pip install pandas numpy scikit-learn rdkit
3) Run:
   python 02_rf_vs_rf_pipeline_anova.py
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# USER SETTINGS
# =========================
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"  # change this if needed
CSV_SEPARATOR = ";"  # use None for automatic delimiter detection

TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"

N_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = -1

MORGAN_BITS = 1024
MORGAN_RADIUS = 2
ADD_RDKIT_DESCRIPTORS = False
USE_FEATURE_CACHE = True

ANOVA_K_VALUES = [50, 100, 150]
RESULTS_CSV = "rf_vs_pipeline_anova_5cv_results.csv"
SELECTED_FEATURES_CSV = "rf_pipeline_selected_features_by_fold.csv"


# =========================
# DATA + FEATURE FUNCTIONS
# =========================
def resolve_input_file(path_like: str | Path) -> Path:
    p = Path(path_like)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path(__file__).resolve().parent / p)
        candidates.append(Path("/mnt/data") / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Input file not found: {path_like}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    if CSV_SEPARATOR is None:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return pd.read_csv(path, sep=CSV_SEPARATOR, encoding="utf-8-sig")


def prepare_binary_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found. Available columns: {list(df.columns)}")

    y_raw = df[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")
    mask = y_num.notna()
    df2 = df.loc[mask].copy()
    y = y_num.loc[mask].astype(int).to_numpy()

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"This script expects binary classification. Found classes: {classes}")
    if not set(classes).issubset({0, 1}):
        mapping = {classes[0]: 0, classes[1]: 1}
        y = np.array([mapping[v] for v in y], dtype=int)
    return df2, y


def smiles_to_features(smiles: list[str], cache_path: Path | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    if USE_FEATURE_CACHE and cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["feature_names"].tolist(), cached["valid_mask"]

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    except ImportError as e:
        raise ImportError(
            "RDKit is required because this dataset contains SMILES rather than ready-made descriptor columns. "
            "Install it with: pip install rdkit"
        ) from e

    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fp_names = [f"morgan_{i}" for i in range(MORGAN_BITS)]

    if ADD_RDKIT_DESCRIPTORS:
        descriptor_names = [name for name, _ in Descriptors._descList]
        descriptor_calc = MolecularDescriptorCalculator(descriptor_names)
    else:
        descriptor_names = []
        descriptor_calc = None

    rows = []
    valid_mask = []

    for i, smi in enumerate(smiles, start=1):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid_mask.append(False)
            continue

        fp = fp_generator.GetFingerprint(mol)
        fp_arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        if descriptor_calc is not None:
            desc = np.array(descriptor_calc.CalcDescriptors(mol), dtype=np.float32)
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
            feat = np.concatenate([fp_arr, desc])
        else:
            feat = fp_arr

        rows.append(feat)
        valid_mask.append(True)

        if i % 500 == 0:
            print(f"Generated features for {i}/{len(smiles)} molecules...")

    X = np.vstack(rows).astype(np.float32)
    feature_names = fp_names + descriptor_names
    valid_mask = np.array(valid_mask, dtype=bool)

    if USE_FEATURE_CACHE and cache_path is not None:
        np.savez_compressed(
            cache_path,
            X=X,
            feature_names=np.array(feature_names, dtype=object),
            valid_mask=valid_mask,
        )

    return X, feature_names, valid_mask


# =========================
# MODEL + EVALUATION
# =========================
def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=80,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def get_models(n_features: int) -> dict[str, object]:
    """Only RF-based designs are included in this file."""
    models: dict[str, object] = {
        # Exactly bare RF: no scaler, no feature selection, no pipeline.
        "RF_raw_all_features_no_scaler_no_FS": make_rf(),

        # A control to show that scaling alone usually does not help tree models.
        # It is still RF, but wrapped in a minimal pipeline for the teaching demo.
        "RF_pipeline_scaler_only_no_ANOVA": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("rf", make_rf()),
            ]
        ),
    }

    for k in ANOVA_K_VALUES:
        k_eff = min(k, n_features)
        models[f"RF_pipeline_scaler_ANOVA_top_{k_eff}"] = Pipeline(
            steps=[
                ("variance", VarianceThreshold(0.0)),
                ("scaler", StandardScaler()),
                ("anova", SelectKBest(score_func=f_classif, k=k_eff)),
                ("rf", make_rf()),
            ]
        )
    return models


def positive_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def extract_selected_features(model, feature_names: list[str]) -> list[str] | None:
    """Extract selected ANOVA features from fitted pipeline, if present."""
    if not isinstance(model, Pipeline):
        return None
    if "anova" not in model.named_steps:
        return None

    names = np.array(feature_names, dtype=object)

    if "variance" in model.named_steps:
        variance_mask = model.named_steps["variance"].get_support()
        names = names[variance_mask]

    anova_mask = model.named_steps["anova"].get_support()
    return names[anova_mask].tolist()


def evaluate_model_cv(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    feature_names: list[str],
) -> tuple[dict[str, float | str], list[dict[str, str | int]]]:
    fold_metrics = []
    selected_feature_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[test_idx])
        y_score = positive_scores(model, X[test_idx])

        metrics = {
            "ROC": roc_auc_score(y[test_idx], y_score),
            "AP": average_precision_score(y[test_idx], y_score),
            "F1": f1_score(y[test_idx], y_pred, zero_division=0),
            "Accuracy": accuracy_score(y[test_idx], y_pred),
            "Recall": recall_score(y[test_idx], y_pred, zero_division=0),
            "Precision": precision_score(y[test_idx], y_pred, zero_division=0),
        }
        fold_metrics.append(metrics)

        selected = extract_selected_features(model, feature_names)
        if selected is not None:
            for rank, feat_name in enumerate(selected, start=1):
                selected_feature_rows.append(
                    {"Model": name, "Fold": fold_idx, "Rank": rank, "Feature": feat_name}
                )

        print(f"  {name} | fold {fold_idx}/{N_SPLITS} done")

    row = {"Model": name}
    for metric_name in ["ROC", "AP", "F1", "Accuracy", "Recall", "Precision"]:
        values = np.array([m[metric_name] for m in fold_metrics], dtype=float)
        row[f"{metric_name}_mean"] = values.mean()
        row[f"{metric_name}_sd"] = values.std(ddof=1)
        row[metric_name] = f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"
    return row, selected_feature_rows


def main() -> None:
    input_path = resolve_input_file(INPUT_FILE)
    print(f"Input file: {input_path}")

    df = read_table(input_path)
    df, y = prepare_binary_target(df, TARGET_COLUMN)

    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"SMILES column '{SMILES_COLUMN}' was not found. Available columns: {list(df.columns)}")

    print(f"Usable binary rows: {len(df)}")
    print("Class counts:")
    print(pd.Series(y).value_counts().sort_index().rename({0: "class 0", 1: "class 1"}))

    cache_path = input_path.with_suffix(f".morgan{MORGAN_BITS}_rdkitdesc_features.npz")
    X, feature_names, valid_mask = smiles_to_features(df[SMILES_COLUMN].tolist(), cache_path=cache_path)
    y = y[valid_mask]

    print(f"Feature matrix: {X.shape[0]} molecules x {X.shape[1]} features")
    print(f"Feature source: Morgan fingerprint ({MORGAN_BITS}) + RDKit descriptors ({len(feature_names) - MORGAN_BITS})")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models(n_features=X.shape[1])

    results = []
    all_selected_features = []
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        try:
            row, selected_rows = evaluate_model_cv(name, model, X, y, cv, feature_names)
            results.append(row)
            all_selected_features.extend(selected_rows)
        except Exception as e:
            print(f"  FAILED: {name} -> {e}")
            results.append({"Model": name, "Error": str(e)})

    results_df = pd.DataFrame(results)

    metric_cols = ["ROC", "AP", "F1", "Accuracy", "Recall", "Precision"]
    display_cols = ["Model"] + [c for c in metric_cols if c in results_df.columns]
    print("\n===== 5-fold CV results, mean ± SD =====")
    print(results_df[display_cols].to_string(index=False))

    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / RESULTS_CSV
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    if all_selected_features:
        selected_df = pd.DataFrame(all_selected_features)
        selected_path = script_dir / SELECTED_FEATURES_CSV
        selected_df.to_csv(selected_path, index=False)
        print(f"Saved selected feature names by fold to: {selected_path}")

    print("\nInterpretation hint for workshop:")
    print(
        "Compare RF_raw_all_features_no_scaler_no_FS against the ANOVA variants. "
        "If raw RF is close to or better than k=50/100/150, that supports the point "
        "that RF can be promising without StandardScaler and without aggressive feature selection."
    )


if __name__ == "__main__":
    main()
