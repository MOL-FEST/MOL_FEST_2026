"""
01_multimodel_classification_benchmark.py

Purpose
-------
Workshop-friendly binary classification benchmark with multiple ML models.
The script reads one input CSV, builds SMILES-derived molecular features
(Morgan fingerprint + RDKit descriptors), runs stratified 5-fold CV, and prints
mean +/- SD for ROC-AUC, Average Precision, F1, Accuracy, Recall, Precision.

How to run
----------
1) Put this .py file and your CSV in the same folder, or edit INPUT_FILE below.
2) Install dependencies if needed:
   pip install pandas numpy scikit-learn rdkit
3) Run:
   python 01_multimodel_classification_benchmark.py

Notes
-----
- For this uploaded dataset, TARGET_COLUMN is binary_label_agonist1_antagonist0.
- Rows with missing target are removed.
- To avoid leakage, assay-result columns, class_code, class_label, etc. are NOT
  used as features. Features are generated only from SMILES.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

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

MORGAN_BITS = 512
MORGAN_RADIUS = 2
ADD_RDKIT_DESCRIPTORS = False
USE_FEATURE_CACHE = True

# Saving outputs
RESULTS_CSV = "multimodel_5cv_results.csv"
INCLUDE_HEAVY_MODELS = False  # set True to also run RBF-SVM and MLP


# =========================
# DATA + FEATURE FUNCTIONS
# =========================
def resolve_input_file(path_like: str | Path) -> Path:
    """Try current folder, script folder, and /mnt/data for convenience."""
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
        # Map two arbitrary numeric labels to 0/1 if necessary.
        mapping = {classes[0]: 0, classes[1]: 1}
        y = np.array([mapping[v] for v in y], dtype=int)
    return df2, y


def smiles_to_features(smiles: list[str], cache_path: Path | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Create Morgan fingerprint + RDKit descriptor features from SMILES.

    Returns
    -------
    X : np.ndarray, shape = [n_valid_molecules, n_features]
    feature_names : list[str]
    valid_mask : np.ndarray bool, True for valid parsed molecules
    """
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
def scaled_model(estimator) -> Pipeline:
    """Variance filter + scaler + estimator. Good for distance/margin/linear models."""
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def tree_model(estimator) -> Pipeline:
    """Variance filter + tree estimator. No scaling because trees do not need it."""
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(0.0)),
            ("model", estimator),
        ]
    )


def get_models() -> dict[str, object]:
    lr = scaled_model(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    )

    # kNN becomes slow/noisy in thousands of fingerprint dimensions.
    # A supervised top-200 ANOVA filter inside CV keeps it workshop-friendly.
    knn = Pipeline(
        steps=[
            ("variance", VarianceThreshold(0.0)),
            ("anova", SelectKBest(score_func=f_classif, k=200)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=N_JOBS)),
        ]
    )

    # Fast linear SVM implementation using SGD hinge loss.
    # This is much faster than LinearSVC/SVC for workshop-scale runs.
    linear_svm = scaled_model(
        SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
    )

    rbf_svm = scaled_model(
        SVC(
            C=3.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_STATE,
        )
    )

    rf = tree_model(
        RandomForestClassifier(
            n_estimators=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    )

    extra = tree_model(
        ExtraTreesClassifier(
            n_estimators=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    )

    gb = tree_model(GradientBoostingClassifier(random_state=RANDOM_STATE))

    hgb = tree_model(
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )
    )

    nb = Pipeline(
        steps=[
            ("variance", VarianceThreshold(0.0)),
            ("model", GaussianNB()),
        ]
    )

    dt = tree_model(
        DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )

    tiny_rf_for_stack = tree_model(
        RandomForestClassifier(
            n_estimators=8,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    )

    mlp = scaled_model(
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=150,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    )

    voting = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("extra", extra),
        ],
        voting="soft",
        n_jobs=1,
    )

    # Fast stacking demo: deliberately lightweight bases so it finishes quickly
    # during a live workshop. Voting above uses the stronger tree ensemble bases.
    stacking = StackingClassifier(
        estimators=[
            ("nb", nb),
            ("dt", dt),
            ("tiny_rf", tiny_rf_for_stack),
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        stack_method="predict_proba",
        cv=2,
        passthrough=False,
        n_jobs=1,
    )

    models = {
        "LogisticRegression": lr,
        "kNN_distance": knn,
        "LinearSVM_SGD_hinge": linear_svm,
        "RandomForest": rf,
        "ExtraTrees": extra,
        "GaussianNB": nb,
        "DecisionTree": dt,
        "SoftVoting_LR_RF_ET": voting,
        "Stacking_NB_DT_tinyRF": stacking,
    }

    # RBF-SVM and MLP are useful for workshop breadth, but can be slower on
    # high-dimensional molecular fingerprints. Enable them from USER SETTINGS
    # when you want a more complete but slower benchmark.
    if INCLUDE_HEAVY_MODELS:
        models["RBF_SVM"] = rbf_svm
        models["GradientBoosting"] = gb
        models["HistGradientBoosting"] = hgb
        models["MLP"] = mlp

    return models


def positive_scores(estimator, X):
    """Return continuous positive-class scores for ROC-AUC/AP."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    # Fallback only for unusual estimators; ROC/AP become less informative.
    return estimator.predict(X)


def evaluate_model_cv(name: str, estimator, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict[str, float | str]:
    fold_metrics = []
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
        print(f"  {name} | fold {fold_idx}/{N_SPLITS} done")

    row = {"Model": name}
    for metric_name in ["ROC", "AP", "F1", "Accuracy", "Recall", "Precision"]:
        values = np.array([m[metric_name] for m in fold_metrics], dtype=float)
        row[f"{metric_name}_mean"] = values.mean()
        row[f"{metric_name}_sd"] = values.std(ddof=1)
        row[metric_name] = f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"
    return row


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
    models = get_models()

    results = []
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        try:
            results.append(evaluate_model_cv(name, model, X, y, cv))
        except Exception as e:
            print(f"  FAILED: {name} -> {e}")
            results.append({"Model": name, "Error": str(e)})

    results_df = pd.DataFrame(results)

    metric_cols = ["ROC", "AP", "F1", "Accuracy", "Recall", "Precision"]
    display_cols = ["Model"] + [c for c in metric_cols if c in results_df.columns]
    print("\n===== 5-fold CV results, mean ± SD =====")
    print(results_df[display_cols].to_string(index=False))

    out_path = Path(__file__).resolve().parent / RESULTS_CSV
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
    main()
