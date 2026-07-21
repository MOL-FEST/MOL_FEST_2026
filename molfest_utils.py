#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOL-FEST gatekeeper pipeline ortak fonksiyonları.

Bu dosya GitHub kök dizinine konur. Notebooklar bu dosyayı indirip kullanır.
02-10 notebookları bağımsız çalışabilsin diye eksik önceki adımları otomatik üretir.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from scipy.stats import randint, loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

DATASETS = {
    "ERa_BLA_assay": {
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv",
        "model_prefix": "model_ERa_BLA",
        "short_name": "ERα BLA",
        "feature_file": "model_ERa_BLA_features.csv",
    },
    "ERa_LUC_VM7_assay": {
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv",
        "model_prefix": "model_ERa_LUC_VM7",
        "short_name": "ERα LUC VM7",
        "feature_file": "model_ERa_LUC_VM7_features.csv",
    },
}

TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MORGAN_BITS = 1024
MORGAN_RADIUS = 2

OUTPUT_ROOT = Path("molfest_outputs")
FEATURE_STORE = Path("molfest_feature_store")
FEATURE_BASE_URL = "https://raw.githubusercontent.com/MOL-FEST/MOL_FEST_2026/main/molfest_feature_store"


# ---------------------------------------------------------------------
# Genel yardımcı fonksiyonlar
# ---------------------------------------------------------------------

def ensure_dirs():
    """Çıktı klasörlerini oluşturur."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT, FEATURE_STORE


def show_table(df, n=50, title=None):
    """Tabloyu Colab'da display ile, terminalde metin olarak gösterir."""
    if title:
        print("\n" + title)
    try:
        display(df.head(n))
    except NameError:
        print(df.head(n).to_string(index=False))


def note(title, message=""):
    """Okunabilir bölüm başlığı basar."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if message:
        print(message)


def percentage_gain(current, previous):
    """Bir metriğin önceki adıma göre yüzde değişimini hesaplar."""
    if previous is None or pd.isna(previous) or abs(previous) < 1e-12:
        return np.nan
    return 100.0 * (current - previous) / abs(previous)


def add_gain_columns(current_df, previous_df, current_step, previous_step):
    """Her veri seti için önceki adımla ROC/AP/F1 farkını ekler."""
    rows = []
    for _, row in current_df.iterrows():
        dataset = row["Dataset"]
        prev_rows = previous_df[previous_df["Dataset"] == dataset]
        out = row.to_dict()
        out["CurrentStep"] = current_step
        out["ComparedWith"] = previous_step
        if prev_rows.empty:
            for metric in ["ROC", "AP", "F1"]:
                out[f"Previous_{metric}"] = np.nan
                out[f"{metric}_Delta"] = np.nan
                out[f"{metric}_Gain_%"] = np.nan
        else:
            prev = prev_rows.sort_values("ROC", ascending=False).iloc[0]
            for metric in ["ROC", "AP", "F1"]:
                out[f"Previous_{metric}"] = prev[metric]
                out[f"{metric}_Delta"] = row[metric] - prev[metric]
                out[f"{metric}_Gain_%"] = percentage_gain(row[metric], prev[metric])
        rows.append(out)
    return pd.DataFrame(rows)


def save_metric_bar(df, metric, title, out_file, label_col):
    """Sonuç tablosunu yatay bar chart olarak kaydeder ve gösterir."""
    plot_df = df.sort_values(metric, ascending=True).copy()
    labels = plot_df[label_col].astype(str).values
    values = plot_df[metric].values
    plt.figure(figsize=(9, max(4, 0.35 * len(plot_df))))
    plt.barh(labels, values)
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


def plot_class_distribution_from_df(df, title, out_file=None):
    """Target sınıf dağılımını bar chart olarak çizer."""
    counts = df["Target"].astype(int).value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar([str(i) for i in counts.index], counts.values)
    plt.xlabel("Sınıf")
    plt.ylabel("Molekül sayısı")
    plt.title(title)
    plt.tight_layout()
    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


def save_progress_plot(df, dataset_key, out_file):
    """Adım adım en iyi ROC değerlerini çizdirir."""
    sub = df[df["Dataset"] == dataset_key].copy()
    plt.figure(figsize=(10, 5))
    plt.plot(sub["ProgressionStep"], sub["ROC"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ROC-AUC")
    plt.title(f"{dataset_key} — adım adım ROC ilerlemesi")
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Veri okuma ve feature üretimi
# ---------------------------------------------------------------------

def github_to_raw(url):
    """GitHub blob linkini raw linke çevirir."""
    return str(url).replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")


def read_csv_flexible(url_or_path):
    """CSV dosyasını önce ';', sonra ',' ayıracıyla okumayı dener."""
    source = github_to_raw(url_or_path)
    last_error = None
    for sep in [";", ","]:
        try:
            df = pd.read_csv(source, sep=sep, encoding="utf-8-sig", low_memory=False)
            if df.shape[1] > 1:
                return df, sep, source
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV okunamadı: {url_or_path}\nSon hata: {last_error}")


def detect_column(df, preferred, keywords):
    """Beklenen kolonu birebir veya anahtar kelimeyle bulur."""
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    raise ValueError(f"Kolon bulunamadı: {preferred}")


def load_raw_dataset(dataset_key):
    """Ham veri setini GitHub'dan okur."""
    info = DATASETS[dataset_key]
    df, sep, source = read_csv_flexible(info["url"])
    target_col = detect_column(df, TARGET_COLUMN, ["binary_label", "label", "target", "class"])
    smiles_col = detect_column(df, SMILES_COLUMN, ["smiles"])
    note(
        f"{info['short_name']} ham verisi okundu",
        f"Kaynak: {source}\nSatır: {df.shape[0]}\nKolon: {df.shape[1]}\nAyraç: {repr(sep)}\nTarget: {target_col}\nSMILES: {smiles_col}"
    )
    return df, target_col, smiles_col, info


def clean_target_and_smiles(df, target_col, smiles_col):
    """Eksik target veya SMILES içeren satırları çıkarır."""
    y_numeric = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_numeric.notna() & df[smiles_col].notna()
    df_clean = df.loc[mask].copy().reset_index(drop=True)
    y = pd.to_numeric(df_clean[target_col], errors="coerce").astype(int).to_numpy()
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Binary target bekleniyor. Bulunan sınıflar: {classes}")
    class_counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()}
    note(
        "Target ve SMILES temizlendi",
        f"Temiz satır: {len(df_clean)}\nÇıkarılan satır: {len(df) - len(df_clean)}\nSınıf dağılımı: {class_counts}"
    )
    return df_clean, y


def valid_molecules(smiles):
    """SMILES listesinden RDKit molecule nesneleri ve geçerli/geçersiz maskesi üretir."""
    mols, valid = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        mols.append(mol)
        valid.append(mol is not None)
    return mols, np.array(valid, dtype=bool)


def smiles_to_morgan(smiles):
    """SMILES listesinden Morgan fingerprint matrisi üretir."""
    mols, valid = valid_molecules(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    names = [f"Morgan_{i}" for i in range(MORGAN_BITS)]
    rows = []
    for mol, keep in zip(mols, valid):
        if keep:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
    return np.vstack(rows), names, valid


def smiles_to_maccs(smiles):
    """SMILES listesinden MACCS fingerprint matrisi üretir."""
    mols, valid = valid_molecules(smiles)
    names = [f"MACCS_{i}" for i in range(1, 167)]
    rows = []
    for mol, keep in zip(mols, valid):
        if keep:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr[1:])
    return np.vstack(rows), names, valid


def smiles_to_rdkit_descriptors(smiles):
    """SMILES listesinden RDKit descriptor matrisi üretir."""
    mols, valid = valid_molecules(smiles)
    descriptor_names = [name for name, _ in Descriptors._descList]
    calc = MolecularDescriptorCalculator(descriptor_names)
    rows = []
    for mol, keep in zip(mols, valid):
        if keep:
            desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
            rows.append(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0))
    names = [f"RDKit_{name}" for name in descriptor_names]
    return np.vstack(rows), names, valid


def smiles_to_avalon(smiles):
    """RDKit kurulumu destekliyorsa Avalon fingerprint üretir."""
    try:
        from rdkit.Avalon import pyAvalonTools
    except Exception:
        print("Avalon desteği bulunamadı; Avalon fingerprint atlandı.")
        return None, [], None
    mols, valid = valid_molecules(smiles)
    names = [f"Avalon_{i}" for i in range(1024)]
    rows = []
    for mol, keep in zip(mols, valid):
        if keep:
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024)
            arr = np.zeros((1024,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
    return np.vstack(rows), names, valid


def generate_feature_table(dataset_key):
    """Bir veri seti için Morgan, MACCS, RDKit ve varsa Avalon feature tablosu üretir."""
    df, target_col, smiles_col, info = load_raw_dataset(dataset_key)
    df_clean, y = clean_target_and_smiles(df, target_col, smiles_col)
    smiles = df_clean[smiles_col].tolist()

    X_morgan, n_morgan, valid_morgan = smiles_to_morgan(smiles)
    X_maccs, n_maccs, valid_maccs = smiles_to_maccs(smiles)
    X_rdkit, n_rdkit, valid_rdkit = smiles_to_rdkit_descriptors(smiles)
    X_avalon, n_avalon, valid_avalon = smiles_to_avalon(smiles)

    if not (np.array_equal(valid_morgan, valid_maccs) and np.array_equal(valid_morgan, valid_rdkit)):
        raise ValueError("Morgan, MACCS ve RDKit valid SMILES maskeleri farklı çıktı.")
    if valid_avalon is not None and not np.array_equal(valid_morgan, valid_avalon):
        raise ValueError("Avalon valid SMILES maskesi diğer feature bloklarından farklı çıktı.")

    valid = valid_morgan
    df_valid = df_clean.loc[valid].reset_index(drop=True)
    y_valid = y[valid]

    blocks = [X_morgan, X_maccs, X_rdkit]
    names = n_morgan + n_maccs + n_rdkit
    if X_avalon is not None:
        blocks.append(X_avalon)
        names += n_avalon

    X_all = np.hstack(blocks)
    feature_df = pd.concat(
        [
            pd.DataFrame({"Dataset": dataset_key, "SMILES": df_valid[smiles_col].values, "Target": y_valid}),
            pd.DataFrame(X_all, columns=names),
        ],
        axis=1,
    )
    return feature_df, names, info


def generate_feature_store():
    """İki veri seti için feature CSV dosyalarını oluşturur."""
    ensure_dirs()
    index_rows = []
    for dataset_key, info in DATASETS.items():
        feature_df, feature_names, info = generate_feature_table(dataset_key)
        out_file = FEATURE_STORE / info["feature_file"]
        feature_df.to_csv(out_file, index=False)
        manifest_file = FEATURE_STORE / f"{info['model_prefix']}_feature_manifest.csv"
        pd.DataFrame({"Feature": feature_names}).to_csv(manifest_file, index=False)
        class_counts = {int(k): int(v) for k, v in feature_df["Target"].value_counts().sort_index().to_dict().items()}
        index_rows.append({
            "Dataset": dataset_key,
            "FeatureFile": str(out_file),
            "ManifestFile": str(manifest_file),
            "Rows": int(feature_df.shape[0]),
            "Columns": int(feature_df.shape[1]),
            "FeatureCount": int(len(feature_names)),
            "Class0": int(class_counts.get(0, 0)),
            "Class1": int(class_counts.get(1, 0)),
            "HasAvalon": bool(any(name.startswith("Avalon_") for name in feature_names)),
        })
        note(
            f"{info['short_name']} feature store hazır",
            f"Dosya: {out_file}\nSatır: {feature_df.shape[0]}\nFeature: {len(feature_names)}"
        )
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(FEATURE_STORE / "feature_store_index.csv", index=False)
    return index_df


def feature_file_path(dataset_key):
    """Feature CSV dosyasını önce local klasörde arar, yoksa GitHub raw linkini döndürür."""
    info = DATASETS[dataset_key]
    local_path = FEATURE_STORE / info["feature_file"]
    if local_path.exists():
        return local_path
    return f"{FEATURE_BASE_URL.rstrip('/')}/{info['feature_file']}"


def read_feature_table(dataset_key):
    """Hazır feature CSV dosyasını okur."""
    path = feature_file_path(dataset_key)
    df = pd.read_csv(path)
    note(
        DATASETS[dataset_key]["short_name"] + " feature tablosu okundu",
        f"Kaynak: {path}\nSatır: {df.shape[0]}\nKolon: {df.shape[1]}"
    )
    return df


def feature_columns(df, feature_set):
    """İstenen feature setine ait kolonları seçer."""
    groups = {
        "morgan": ["Morgan_"],
        "maccs": ["MACCS_"],
        "rdkit": ["RDKit_"],
        "avalon": ["Avalon_"],
        "maccs_morgan": ["MACCS_", "Morgan_"],
        "maccs_rdkit": ["MACCS_", "RDKit_"],
        "morgan_rdkit": ["Morgan_", "RDKit_"],
        "all_available": ["Morgan_", "MACCS_", "RDKit_", "Avalon_"],
    }
    if feature_set not in groups:
        raise ValueError(f"Bilinmeyen feature_set: {feature_set}")
    cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in groups[feature_set])]
    if not cols:
        raise ValueError(f"{feature_set} için feature kolonu bulunamadı.")
    return cols


def available_feature_sets(df):
    """Feature tablosunda kullanılabilecek feature setlerini listeler."""
    sets = ["morgan", "maccs", "rdkit", "maccs_morgan", "maccs_rdkit", "morgan_rdkit", "all_available"]
    if any(c.startswith("Avalon_") for c in df.columns):
        sets.insert(3, "avalon")
    return sets


# ---------------------------------------------------------------------
# Modelleme yardımcıları
# ---------------------------------------------------------------------

def split_xy(df, cols):
    """Feature kolonlarından X matrisi, Target kolonundan y vektörü üretir ve stratified split yapar."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["Target"].astype(int).to_numpy()
    return train_test_split(X, y, df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def get_score_class1(model, X):
    """Modelden class 1 için skor/olasılık üretir."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def calculate_metrics(y_true, y_pred, y_score):
    """Sınıflandırma metriklerini hesaplar."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "ROC": roc_auc_score(y_true, y_score),
        "AP": average_precision_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def make_model(model_type, random_state=RANDOM_STATE):
    """Model adından sklearn model nesnesi üretir."""
    if model_type == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    if model_type == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=random_state, n_jobs=-1)
    if model_type == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=random_state)
    if model_type == "LogisticRegression":
        return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))])
    if model_type == "kNN":
        return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))])
    if model_type == "LinearSVM":
        return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=random_state), cv=3))])
    if model_type == "RBF_SVM":
        return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=random_state))])
    if model_type == "GradientBoosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_type == "GaussianNB":
        return GaussianNB()
    if model_type == "DecisionTree":
        return DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=random_state)
    raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def ten_model_types():
    """10 model benchmark listesini verir."""
    return ["LogisticRegression", "kNN", "LinearSVM", "RBF_SVM", "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "GaussianNB", "DecisionTree"]


def candidate_model_types():
    """Pipeline boyunca taşınacak ana ağaç tabanlı aday modelleri verir."""
    return ["RandomForest", "ExtraTrees", "HistGradientBoosting"]


def fit_evaluate(model, X_train, X_test, y_train, y_test):
    """Modeli eğitir, test tahmini üretir ve metrikleri hesaplar."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = get_score_class1(model, X_test)
    metrics = calculate_metrics(y_test, y_pred, y_score)
    return model, y_pred, y_score, metrics


def save_model(model, feature_names, path, extra=None):
    """Modeli, kullanılan feature isimleriyle birlikte kaydeder."""
    payload = {"model": model, "feature_names": feature_names}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def save_predictions(df_test, y_test, y_pred, y_score, path):
    """Test tahminlerini CSV olarak kaydeder."""
    pred = pd.DataFrame({
        "SMILES": df_test["SMILES"].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score_class1": y_score,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(path, index=False)
    return pred


def write_feature_list(features, path):
    """Seçilen feature isimlerini txt dosyasına yazar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(features), encoding="utf-8")


def read_feature_list(path):
    """Txt dosyasındaki feature isimlerini okur."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_feature_ranking(method, X_train, y_train, feature_names):
    """Seçilen yönteme göre feature sıralaması üretir."""
    if method == "none":
        return list(range(len(feature_names)))
    if method == "ANOVA":
        scores, _ = f_classif(X_train, y_train)
        return np.argsort(np.nan_to_num(scores))[::-1].tolist()
    if method == "Chi2":
        X_chi = X_train
        if np.nanmin(X_chi) < 0:
            X_chi = MinMaxScaler().fit_transform(X_chi)
        scores, _ = chi2(X_chi, y_train)
        return np.argsort(np.nan_to_num(scores))[::-1].tolist()
    if method == "MutualInfo":
        scores = mutual_info_classif(X_train, y_train, discrete_features="auto", random_state=RANDOM_STATE)
        return np.argsort(np.nan_to_num(scores))[::-1].tolist()
    if method == "RF_importance":
        selector = make_model("RandomForest")
        selector.fit(X_train, y_train)
        return np.argsort(selector.feature_importances_)[::-1].tolist()
    raise ValueError(f"Bilinmeyen feature selection yöntemi: {method}")


def apply_resampling(y_train, ratio=1.0, method="none"):
    """Train set için over/under sampling indexleri üretir."""
    if method == "none":
        return np.arange(len(y_train))
    rng = np.random.RandomState(RANDOM_STATE)
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train == 0)[0]
    current = len(pos) / len(neg)

    if method == "oversampling":
        if current < ratio:
            n_pos, n_neg = int(np.ceil(ratio * len(neg))), len(neg)
        else:
            n_pos, n_neg = len(pos), int(np.ceil(len(pos) / ratio))
        s_pos = rng.choice(pos, n_pos, replace=n_pos > len(pos))
        s_neg = rng.choice(neg, n_neg, replace=n_neg > len(neg))
    elif method == "undersampling":
        if current < ratio:
            n_pos = len(pos)
            n_neg = min(len(neg), max(5, int(np.floor(len(pos) / ratio))))
        else:
            n_pos = min(len(pos), max(5, int(np.floor(ratio * len(neg)))))
            n_neg = len(neg)
        s_pos = rng.choice(pos, n_pos, replace=False)
        s_neg = rng.choice(neg, n_neg, replace=False)
    else:
        raise ValueError("method none, oversampling veya undersampling olmalı.")

    idx = np.concatenate([s_pos, s_neg])
    rng.shuffle(idx)
    return idx


def param_space(model_type):
    """Tuning yapılacak model için hiperparametre aralığını verir."""
    if model_type == "RandomForest":
        return {"n_estimators": randint(150, 500), "max_depth": [None, 5, 10, 20, 40], "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", "log2", None]}
    if model_type == "ExtraTrees":
        return {"n_estimators": randint(150, 500), "max_depth": [None, 5, 10, 20, 40], "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", "log2", None]}
    if model_type == "HistGradientBoosting":
        return {"max_iter": randint(80, 220), "learning_rate": loguniform(0.02, 0.2), "max_leaf_nodes": randint(15, 60), "l2_regularization": loguniform(1e-5, 1.0)}
    raise ValueError(f"Bu model tipi için tuning alanı tanımlı değil: {model_type}")


# ---------------------------------------------------------------------
# Pipeline adımları
# ---------------------------------------------------------------------

def run_step_01_feature_store():
    """Feature CSV dosyalarını üretir ve sınıf dağılımı grafiklerini çizer."""
    lesson_out = OUTPUT_ROOT / "01_feature_store"
    lesson_out.mkdir(parents=True, exist_ok=True)
    feature_index = generate_feature_store()
    feature_index.to_csv(lesson_out / "feature_store_index_copy.csv", index=False)

    for dataset_key, info in DATASETS.items():
        feature_file = FEATURE_STORE / info["feature_file"]
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            plot_class_distribution_from_df(df, f"{info['short_name']} sınıf dağılımı", lesson_out / f"{info['model_prefix']}_class_distribution.png")
    show_table(feature_index, title="Feature store özeti")
    return feature_index


def run_step_02_baseline():
    """RF + Morgan baseline modelini çalıştırır."""
    ensure_dirs()
    lesson_out = OUTPUT_ROOT / "02_rf_morgan_baseline"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        plot_class_distribution_from_df(df, f"{DATASETS[dataset_key]['short_name']} sınıf dağılımı", lesson_out / f"{DATASETS[dataset_key]['model_prefix']}_class_distribution.png")

        cols = feature_columns(df, "morgan")
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, cols)

        model = make_model("RandomForest")
        model, y_pred, y_score, metrics = fit_evaluate(model, X_train, X_test, y_train, y_test)

        model_name = f"{DATASETS[dataset_key]['model_prefix']}_RF_morgan_baseline"
        metrics.update({"Dataset": dataset_key, "Step": "02_baseline", "ModelType": "RandomForest", "FeatureSet": "morgan", "SelectionMethod": "none", "K": len(cols), "ModelName": model_name, "FeatureListFile": ""})
        rows.append(metrics)

        save_model(model, cols, models_dir / f"{model_name}.joblib", extra=metrics)
        save_predictions(df_test, y_test, y_pred, y_score, lesson_out / f"{model_name}_predictions.csv")

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["class 0", "class 1"])
        plt.title(f"{model_name} — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(lesson_out / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()

        RocCurveDisplay.from_predictions(y_test, y_score)
        plt.title(f"{model_name} — ROC Curve")
        plt.tight_layout()
        plt.savefig(lesson_out / f"{model_name}_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

    baseline_df = pd.DataFrame(rows).sort_values("ROC", ascending=False)
    baseline_df.to_csv(lesson_out / "02_rf_morgan_baseline_metrics.csv", index=False)
    show_table(baseline_df, title="02 — RF + Morgan baseline sonuçları")
    return baseline_df


def ensure_step_02():
    path = OUTPUT_ROOT / "02_rf_morgan_baseline" / "02_rf_morgan_baseline_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    note("02 sonucu bulunamadı", "RF + Morgan baseline otomatik çalıştırılıyor.")
    return run_step_02_baseline()


def run_step_03_feature_ablation():
    """RF sabitken feature set karşılaştırması yapar."""
    previous_df = ensure_step_02()
    lesson_out = OUTPUT_ROOT / "03_rf_feature_ablation"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows, best_rows = [], []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        dataset_rows = []
        for fset in available_feature_sets(df):
            cols = feature_columns(df, fset)
            X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, cols)
            model = make_model("RandomForest")
            model, y_pred, y_score, metrics = fit_evaluate(model, X_train, X_test, y_train, y_test)
            model_name = f"{DATASETS[dataset_key]['model_prefix']}_RF_{fset}"
            metrics.update({"Dataset": dataset_key, "Step": "03_feature_ablation", "ModelType": "RandomForest", "FeatureSet": fset, "SelectionMethod": "none", "K": len(cols), "n_features": len(cols), "ModelName": model_name, "FeatureListFile": ""})
            dataset_rows.append(metrics)
            all_rows.append(metrics)
            save_model(model, cols, models_dir / f"{model_name}.joblib", extra=metrics)

        dataset_df = pd.DataFrame(dataset_rows).sort_values("ROC", ascending=False)
        dataset_df.to_csv(lesson_out / f"{dataset_key}_feature_ablation.csv", index=False)
        save_metric_bar(dataset_df, "ROC", f"{dataset_key} — feature ablation ROC", lesson_out / f"{dataset_key}_feature_ablation_roc.png", label_col="FeatureSet")
        show_table(dataset_df, title=f"{dataset_key} feature ablation sonuçları")
        best_rows.append(dataset_df.iloc[0].to_dict())

    ablation_df = pd.DataFrame(all_rows).sort_values(["Dataset", "ROC"], ascending=[True, False])
    ablation_df.to_csv(lesson_out / "03_feature_ablation_all.csv", index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "03_best_feature_set_per_dataset.csv", index=False)

    gain_df = add_gain_columns(best_df, previous_df, "03_Feature_Ablation", "02_RF_Morgan_Baseline")
    gain_df.to_csv(lesson_out / "03_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "FeatureSet", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "AP", "F1"]], title="03 — bir önceki adıma göre kazanç")
    return best_df


def ensure_step_03():
    path = OUTPUT_ROOT / "03_rf_feature_ablation" / "03_best_feature_set_per_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    note("03 sonucu bulunamadı", "Feature ablation otomatik çalıştırılıyor.")
    return run_step_03_feature_ablation()


def run_step_04_feature_selection():
    """03'te seçilen feature set üzerinde top-k feature selection yapar."""
    previous_df = ensure_step_03()
    lesson_out = OUTPUT_ROOT / "04_feature_selection"
    selected_dir = lesson_out / "selected_features"
    selected_dir.mkdir(parents=True, exist_ok=True)

    methods = ["none", "ANOVA", "Chi2", "MutualInfo", "RF_importance"]
    k_values = [50, 100, 150, 200]
    all_rows, best_rows = [], []

    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        feature_set = previous_df.loc[previous_df["Dataset"] == dataset_key, "FeatureSet"].iloc[0]
        base_cols = feature_columns(df, feature_set)
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, base_cols)
        dataset_rows = []

        for method in methods:
            ranking = select_feature_ranking(method, X_train, y_train, base_cols)
            current_k_values = [len(base_cols)] if method == "none" else [min(k, len(base_cols)) for k in k_values]
            for k in current_k_values:
                idx = ranking[:k]
                selected_cols = [base_cols[i] for i in idx]
                feature_file = selected_dir / f"{dataset_key}_{method}_top{k}.txt"
                write_feature_list(selected_cols, feature_file)

                model = make_model("RandomForest")
                model, y_pred, y_score, metrics = fit_evaluate(model, X_train[:, idx], X_test[:, idx], y_train, y_test)
                metrics.update({"Dataset": dataset_key, "Step": "04_feature_selection", "ModelType": "RandomForest", "FeatureSet": feature_set, "SelectionMethod": method, "K": k, "FeatureListFile": str(feature_file)})
                dataset_rows.append(metrics)
                all_rows.append(metrics)

        dataset_df = pd.DataFrame(dataset_rows).sort_values("ROC", ascending=False)
        dataset_df.to_csv(lesson_out / f"{dataset_key}_feature_selection.csv", index=False)
        save_metric_bar(dataset_df.head(15), "ROC", f"{dataset_key} — feature selection ROC", lesson_out / f"{dataset_key}_feature_selection_top15_roc.png", label_col="SelectionMethod")
        show_table(dataset_df, title=f"{dataset_key} feature selection sonuçları")
        best_rows.append(dataset_df.iloc[0].to_dict())

    result_df = pd.DataFrame(all_rows).sort_values(["Dataset", "ROC"], ascending=[True, False])
    result_df.to_csv(lesson_out / "04_feature_selection_all.csv", index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "04_best_feature_selection_per_dataset.csv", index=False)

    gain_df = add_gain_columns(best_df, previous_df, "04_Feature_Selection", "03_Feature_Ablation")
    gain_df.to_csv(lesson_out / "04_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "SelectionMethod", "K", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "AP", "F1"]], title="04 — bir önceki adıma göre kazanç")
    return best_df


def ensure_step_04():
    path = OUTPUT_ROOT / "04_feature_selection" / "04_best_feature_selection_per_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    note("04 sonucu bulunamadı", "Feature selection otomatik çalıştırılıyor.")
    return run_step_04_feature_selection()


def run_step_05_train_10_models():
    """Seçilen featurelarla 10 model karşılaştırması yapar."""
    previous_df = ensure_step_04()
    lesson_out = OUTPUT_ROOT / "05_train_10_models"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows, carry_rows, best_rows = [], [], []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        row = previous_df[previous_df["Dataset"] == dataset_key].iloc[0]
        selected_cols = read_feature_list(row["FeatureListFile"])
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, selected_cols)
        dataset_rows = []

        for model_type in ten_model_types():
            model = make_model(model_type)
            model, y_pred, y_score, metrics = fit_evaluate(model, X_train, X_test, y_train, y_test)
            model_name = f"{DATASETS[dataset_key]['model_prefix']}_{model_type}_selected_features"
            metrics.update({"Dataset": dataset_key, "Step": "05_10_models", "ModelType": model_type, "FeatureSet": row.get("FeatureSet", ""), "SelectionMethod": row.get("SelectionMethod", ""), "K": row.get("K", len(selected_cols)), "ModelName": model_name, "FeatureListFile": row["FeatureListFile"]})
            dataset_rows.append(metrics)
            all_rows.append(metrics)
            save_model(model, selected_cols, models_dir / f"{model_name}.joblib", extra=metrics)

        dataset_df = pd.DataFrame(dataset_rows).sort_values("ROC", ascending=False)
        dataset_df.to_csv(lesson_out / f"{dataset_key}_10_model_metrics.csv", index=False)
        save_metric_bar(dataset_df, "ROC", f"{dataset_key} — 10 model ROC", lesson_out / f"{dataset_key}_10_model_roc.png", label_col="ModelType")
        show_table(dataset_df, title=f"{dataset_key} 10 model karşılaştırması")

        tree_df = dataset_df[dataset_df["ModelType"].isin(candidate_model_types())].sort_values("ROC", ascending=False).copy()
        tree_df.to_csv(lesson_out / f"{dataset_key}_tree_candidate_models.csv", index=False)
        carry_rows.append(tree_df)
        best_rows.append(dataset_df.iloc[0].to_dict())

    all_df = pd.DataFrame(all_rows).sort_values(["Dataset", "ROC"], ascending=[True, False])
    all_df.to_csv(lesson_out / "05_10_model_metrics_all.csv", index=False)
    carry_df = pd.concat(carry_rows, ignore_index=True)
    carry_df.to_csv(lesson_out / "05_top3_candidate_models.csv", index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "05_best_model_per_dataset.csv", index=False)

    gain_df = add_gain_columns(best_df, previous_df, "05_10_Model_Search", "04_Feature_Selection")
    gain_df.to_csv(lesson_out / "05_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "ModelType", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "AP", "F1"]], title="05 — bir önceki adıma göre kazanç")
    show_table(carry_df[["Dataset", "ModelType", "ROC", "AP", "F1", "FeatureListFile"]], title="Sonraki adıma taşınacak ağaç tabanlı adaylar")
    return carry_df


def ensure_step_05():
    path = OUTPUT_ROOT / "05_train_10_models" / "05_top3_candidate_models.csv"
    if path.exists():
        return pd.read_csv(path)
    note("05 sonucu bulunamadı", "10 model arama otomatik çalıştırılıyor.")
    return run_step_05_train_10_models()


def run_step_06_resampling():
    """Üç ana aday model için sampling senaryolarını dener."""
    candidates_df = ensure_step_05()
    previous_df = candidates_df.sort_values(["Dataset", "ROC"], ascending=[True, False]).groupby("Dataset").head(1).copy()

    lesson_out = OUTPUT_ROOT / "06_resampling"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("none", 1.0, "none"),
        ("balanced_oversampling", 1.0, "oversampling"),
        ("balanced_undersampling", 1.0, "undersampling"),
        ("positive_5x_oversampling", 5.0, "oversampling"),
        ("negative_5x_oversampling", 0.2, "oversampling"),
        ("positive_5x_undersampling", 5.0, "undersampling"),
        ("negative_5x_undersampling", 0.2, "undersampling"),
    ]

    all_rows, best_rows = [], []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        candidates = candidates_df[candidates_df["Dataset"] == dataset_key]
        selected_cols = read_feature_list(candidates.iloc[0]["FeatureListFile"])
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, selected_cols)
        dataset_rows = []

        for _, cand in candidates.iterrows():
            model_type = cand["ModelType"]
            for scenario_name, ratio, method in scenarios:
                idx = apply_resampling(y_train, ratio=ratio, method=method)
                X_res, y_res = X_train[idx], y_train[idx]
                model = make_model(model_type)
                model, y_pred, y_score, metrics = fit_evaluate(model, X_res, X_test, y_res, y_test)
                model_name = f"{DATASETS[dataset_key]['model_prefix']}_{model_type}_{scenario_name}"
                metrics.update({"Dataset": dataset_key, "Step": "06_resampling", "ModelType": model_type, "SamplingScenario": scenario_name, "SamplingMethod": method, "SamplingRatio": ratio, "FeatureListFile": candidates.iloc[0]["FeatureListFile"], "ModelName": model_name})
                dataset_rows.append(metrics)
                all_rows.append(metrics)
                save_model(model, selected_cols, models_dir / f"{model_name}.joblib", extra=metrics)

        dataset_df = pd.DataFrame(dataset_rows).sort_values("ROC", ascending=False)
        dataset_df.to_csv(lesson_out / f"{dataset_key}_resampling_metrics.csv", index=False)
        save_metric_bar(dataset_df.head(20), "ROC", f"{dataset_key} — sampling ROC", lesson_out / f"{dataset_key}_resampling_top20_roc.png", label_col="SamplingScenario")
        show_table(dataset_df, title=f"{dataset_key} sampling sonuçları")
        best_rows.append(dataset_df.iloc[0].to_dict())

    all_df = pd.DataFrame(all_rows).sort_values(["Dataset", "ROC"], ascending=[True, False])
    all_df.to_csv(lesson_out / "06_resampling_metrics_all.csv", index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "06_best_sampling_per_dataset.csv", index=False)

    gain_df = add_gain_columns(best_df, previous_df, "06_Resampling", "05_Tree_Candidate_Search")
    gain_df.to_csv(lesson_out / "06_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "ModelType", "SamplingScenario", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "AP", "F1"]], title="06 — bir önceki aday modele göre kazanç")
    return best_df


def ensure_step_06():
    path = OUTPUT_ROOT / "06_resampling" / "06_best_sampling_per_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    note("06 sonucu bulunamadı", "Sampling adımı otomatik çalıştırılıyor.")
    return run_step_06_resampling()


def run_step_07_tuning():
    """06'da seçilen en iyi model/sampling kombinasyonu üzerinde random search tuning yapar."""
    previous_df = ensure_step_06()
    lesson_out = OUTPUT_ROOT / "07_random_search_tuning"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        best = previous_df[previous_df["Dataset"] == dataset_key].iloc[0]
        selected_cols = read_feature_list(best["FeatureListFile"])
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, selected_cols)
        idx = apply_resampling(y_train, ratio=float(best["SamplingRatio"]), method=best["SamplingMethod"])
        X_res, y_res = X_train[idx], y_train[idx]

        model_type = best["ModelType"]
        search = RandomizedSearchCV(
            estimator=make_model(model_type),
            param_distributions=param_space(model_type),
            n_iter=10,
            scoring="roc_auc",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        note(f"Random search tuning: {dataset_key} | {model_type}")
        search.fit(X_res, y_res)

        tuned_model = search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        y_score = get_score_class1(tuned_model, X_test)
        metrics = calculate_metrics(y_test, y_pred, y_score)

        model_name = f"{DATASETS[dataset_key]['model_prefix']}_{model_type}_tuned"
        metrics.update({"Dataset": dataset_key, "Step": "07_tuning", "ModelType": model_type, "FeatureListFile": best["FeatureListFile"], "SamplingScenario": best["SamplingScenario"], "SamplingMethod": best["SamplingMethod"], "SamplingRatio": best["SamplingRatio"], "BestCV_ROC": search.best_score_, "BestParams": str(search.best_params_), "ModelName": model_name})
        rows.append(metrics)
        save_model(tuned_model, selected_cols, models_dir / f"{model_name}.joblib", extra=metrics)

    tuning_df = pd.DataFrame(rows).sort_values("ROC", ascending=False)
    tuning_df.to_csv(lesson_out / "07_tuning_metrics.csv", index=False)

    gain_df = add_gain_columns(tuning_df, previous_df, "07_Tuning", "06_Resampling")
    gain_df.to_csv(lesson_out / "07_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "ModelType", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "BestCV_ROC", "AP", "F1"]], title="07 — bir önceki adıma göre kazanç")
    return tuning_df


def ensure_step_07():
    path = OUTPUT_ROOT / "07_random_search_tuning" / "07_tuning_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    note("07 sonucu bulunamadı", "Tuning adımı otomatik çalıştırılıyor.")
    return run_step_07_tuning()


def run_step_08_ensemble():
    """Seçilmiş feature alanında ensemble ve stacking modellerini dener."""
    previous_df = ensure_step_07()
    selection_df = ensure_step_04()

    lesson_out = OUTPUT_ROOT / "08_ensemble_stacking"
    models_dir = lesson_out / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    def ensemble_candidates():
        rf = make_model("RandomForest")
        et = make_model("ExtraTrees")
        hgb = make_model("HistGradientBoosting")
        voting = VotingClassifier([("rf", rf), ("et", et), ("hgb", hgb)], voting="soft")
        stacking = StackingClassifier(
            [("rf", rf), ("et", et), ("hgb", hgb)],
            final_estimator=LogisticRegression(max_iter=2000, solver="liblinear"),
            stack_method="predict_proba",
            cv=3,
        )
        return {"RandomForest": rf, "ExtraTrees": et, "HistGradientBoosting": hgb, "SoftVoting": voting, "Stacking": stacking}

    rows, best_rows = [], []
    for dataset_key in DATASETS:
        df = read_feature_table(dataset_key)
        best_sel = selection_df[selection_df["Dataset"] == dataset_key].iloc[0]
        selected_cols = read_feature_list(best_sel["FeatureListFile"])
        X_train, X_test, y_train, y_test, df_train, df_test = split_xy(df, selected_cols)

        dataset_rows = []
        for model_type, model in ensemble_candidates().items():
            model, y_pred, y_score, metrics = fit_evaluate(model, X_train, X_test, y_train, y_test)
            model_name = f"{DATASETS[dataset_key]['model_prefix']}_{model_type}_ensemble_step"
            metrics.update({"Dataset": dataset_key, "Step": "08_ensemble", "ModelType": model_type, "FeatureListFile": best_sel["FeatureListFile"], "ModelName": model_name})
            dataset_rows.append(metrics)
            rows.append(metrics)
            save_model(model, selected_cols, models_dir / f"{model_name}.joblib", extra=metrics)

        dataset_df = pd.DataFrame(dataset_rows).sort_values("ROC", ascending=False)
        dataset_df.to_csv(lesson_out / f"{dataset_key}_ensemble_metrics.csv", index=False)
        save_metric_bar(dataset_df, "ROC", f"{dataset_key} — ensemble ROC", lesson_out / f"{dataset_key}_ensemble_roc.png", label_col="ModelType")
        show_table(dataset_df, title=f"{dataset_key} ensemble / stacking sonuçları")
        best_rows.append(dataset_df.iloc[0].to_dict())

    ensemble_df = pd.DataFrame(rows).sort_values(["Dataset", "ROC"], ascending=[True, False])
    ensemble_df.to_csv(lesson_out / "08_ensemble_metrics_all.csv", index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "08_best_ensemble_per_dataset.csv", index=False)

    gain_df = add_gain_columns(best_df, previous_df, "08_Ensemble_Stacking", "07_Tuning")
    gain_df.to_csv(lesson_out / "08_gain_vs_previous.csv", index=False)
    show_table(gain_df[["Dataset", "ModelType", "ROC", "Previous_ROC", "ROC_Delta", "ROC_Gain_%", "AP", "F1"]], title="08 — tuning adımına göre kazanç")
    return best_df


def ensure_step_08():
    path = OUTPUT_ROOT / "08_ensemble_stacking" / "08_best_ensemble_per_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    note("08 sonucu bulunamadı", "Ensemble/stacking adımı otomatik çalıştırılıyor.")
    return run_step_08_ensemble()


def run_step_09_collect_best_candidates():
    """02-08 adımlarındaki en iyi adayları tek tabloda toplar."""
    ensure_step_02()
    ensure_step_03()
    ensure_step_04()
    ensure_step_05()
    ensure_step_06()
    ensure_step_07()
    ensure_step_08()

    lesson_out = OUTPUT_ROOT / "09_collect_best_candidates"
    lesson_out.mkdir(parents=True, exist_ok=True)

    step_files = {
        "02_RF_Morgan_Baseline": OUTPUT_ROOT / "02_rf_morgan_baseline" / "02_rf_morgan_baseline_metrics.csv",
        "03_Feature_Ablation": OUTPUT_ROOT / "03_rf_feature_ablation" / "03_feature_ablation_all.csv",
        "04_Feature_Selection": OUTPUT_ROOT / "04_feature_selection" / "04_feature_selection_all.csv",
        "05_10_Model_Search": OUTPUT_ROOT / "05_train_10_models" / "05_10_model_metrics_all.csv",
        "06_Resampling": OUTPUT_ROOT / "06_resampling" / "06_resampling_metrics_all.csv",
        "07_Tuning": OUTPUT_ROOT / "07_random_search_tuning" / "07_tuning_metrics.csv",
        "08_Ensemble_Stacking": OUTPUT_ROOT / "08_ensemble_stacking" / "08_ensemble_metrics_all.csv",
    }

    best_rows = []
    for dataset_key in DATASETS:
        for step_name, path in step_files.items():
            df = pd.read_csv(path)
            df = df[df["Dataset"] == dataset_key].copy()
            if df.empty:
                continue
            best = df.sort_values("ROC", ascending=False).iloc[0].to_dict()
            best["ProgressionStep"] = step_name
            best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(lesson_out / "09_best_candidates_by_step.csv", index=False)

    summary_rows = []
    for dataset_key in DATASETS:
        sub = best_df[best_df["Dataset"] == dataset_key].copy()
        baseline = sub[sub["ProgressionStep"] == "02_RF_Morgan_Baseline"].iloc[0]
        best = sub.sort_values("ROC", ascending=False).iloc[0]
        summary_rows.append({
            "Dataset": dataset_key,
            "BaselineStep": baseline["ProgressionStep"],
            "BaselineModel": baseline.get("ModelType", ""),
            "Baseline_ROC": baseline["ROC"],
            "Baseline_AP": baseline["AP"],
            "Baseline_F1": baseline["F1"],
            "BestStep": best["ProgressionStep"],
            "BestModel": best.get("ModelType", ""),
            "Best_ROC": best["ROC"],
            "Best_AP": best["AP"],
            "Best_F1": best["F1"],
            "ROC_Gain": best["ROC"] - baseline["ROC"],
            "ROC_Gain_%": percentage_gain(best["ROC"], baseline["ROC"]),
            "AP_Gain": best["AP"] - baseline["AP"],
            "F1_Gain": best["F1"] - baseline["F1"],
        })
        show_table(sub[["ProgressionStep", "ModelType", "ROC", "AP", "F1", "Accuracy", "BalancedAccuracy"]], title=f"{dataset_key} için adım bazlı en iyi adaylar")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(lesson_out / "09_best_candidates_summary.csv", index=False)
    show_table(summary_df, title="Baseline RF Morgan ile en iyi aday karşılaştırması")
    return best_df, summary_df


def run_step_10_final_report():
    """Final progression raporunu üretir."""
    best_df, summary_df = run_step_09_collect_best_candidates()

    lesson_out = OUTPUT_ROOT / "10_final_progression_report"
    lesson_out.mkdir(parents=True, exist_ok=True)

    for dataset_key in DATASETS:
        sub = best_df[best_df["Dataset"] == dataset_key].copy()
        sub.to_csv(lesson_out / f"{dataset_key}_progression.csv", index=False)
        show_table(sub[["ProgressionStep", "ModelType", "ROC", "AP", "F1", "Accuracy", "BalancedAccuracy"]], title=f"{dataset_key} için adım adım en iyi sonuçlar")
        save_progress_plot(sub, dataset_key, lesson_out / f"{dataset_key}_progression_roc.png")

    summary_df.to_csv(lesson_out / "10_baseline_vs_best_final_summary.csv", index=False)
    show_table(summary_df, title="Baseline RF Morgan ile en iyi final sonuç karşılaştırması")
    return summary_df
