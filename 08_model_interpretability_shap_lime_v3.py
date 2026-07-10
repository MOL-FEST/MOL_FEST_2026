#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ders 4B — SHAP, SHAP Waterfall ve LIME

# %% [markdown]
# # Ders 4B — SHAP, SHAP Waterfall ve LIME
# 
# İki veri seti için ayrı Random Forest modelleri eğitilir; SHAP beeswarm, SHAP bar, SHAP waterfall ve LIME çıktıları üretilir.

# %% [markdown]
# ## 1. Paket kurulumu
# 
# Bu hücre gerekli Python paketlerini kontrol eder ve eksik olanları kurar.  
# RDKit, SMILES metninden moleküler fingerprint üretmek için kullanılır.

# %%
import sys
import subprocess
import importlib.util

def install_if_missing(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        print(f"[INSTALL] {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

required_packages = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("joblib", "joblib"),
]

for import_name, pip_name in required_packages:
    install_if_missing(import_name, pip_name)

if importlib.util.find_spec("rdkit") is None:
    try:
        install_if_missing("rdkit", "rdkit")
    except Exception:
        install_if_missing("rdkit", "rdkit-pypi")

print("Paket kontrolü tamamlandı.")

# %% [markdown]
# ## 2. Paketleri çağırma
# 
# Bu hücre analizde kullanılacak paketleri çağırır.  
# Kurulum ve paket çağırma işlemleri ayrı tutulur; böylece hata olduğunda hangi aşamada sorun çıktığı daha kolay anlaşılır.

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

print("Importlar tamamlandı.")

# %% [markdown]
# ## 3. Genel ayarlar
# 
# Bu hücre veri linklerini, hedef kolonu, SMILES kolonunu, modelleme sabitlerini ve çıktı klasörünü tanımlar.  
# İki veri seti de aynı notebook içinde ayrı ayrı işlenir. Model isimleri karışmaması için şu ön ekler kullanılır:
# 
# - `model_ERa_BLA`
# - `model_ERa_LUC_VM7`

# %%
DATASETS = {
    "ERa_BLA_assay": {
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv",
        "model_prefix": "model_ERa_BLA",
        "short_name": "ERα BLA"
    },
    "ERa_LUC_VM7_assay": {
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv",
        "model_prefix": "model_ERa_LUC_VM7",
        "short_name": "ERα LUC VM7"
    },
}

TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MORGAN_BITS = 1024
MORGAN_RADIUS = 2

OUTPUT_ROOT = Path("molfest_outputs_v3")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Genel ayarlar hazır.")
print("Çıktı klasörü:", OUTPUT_ROOT.resolve())
print("Veri setleri:", list(DATASETS.keys()))

# %% [markdown]
# ## 4. Veri okuma fonksiyonları
# 
# Bu bölümün görevi veriyi güvenli şekilde okumaktır.
# 
# Yapılan işlemler:
# - GitHub `blob/main` linki raw CSV linkine çevrilir.
# - CSV ayıracı `;` veya `,` olarak denenir.
# - Target kolonu bulunur.
# - SMILES kolonu bulunur.
# - Eksik target veya eksik SMILES içeren satırlar çıkarılır.
# - Satır, kolon ve sınıf dağılımı bilgileri ekrana yazdırılır.

# %%
def ensure_output_root():
    global OUTPUT_ROOT
    if "OUTPUT_ROOT" not in globals():
        OUTPUT_ROOT = Path("molfest_outputs_v3")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT

def note(title, message=""):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if message:
        print(message)

def github_to_raw(url):
    return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

def read_csv_flexible(url):
    source = github_to_raw(url)
    for sep in [";", ","]:
        df = pd.read_csv(source, sep=sep, encoding="utf-8-sig", low_memory=False)
        if df.shape[1] > 1:
            return df, sep, source
    raise ValueError("CSV okunamadı. Link veya ayraç kontrol edilmeli.")

def detect_column(df, preferred, keywords):
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    raise ValueError(f"Kolon bulunamadı: {preferred}")

def load_one_dataset(dataset_key):
    info = DATASETS[dataset_key]
    df, sep, source = read_csv_flexible(info["url"])
    target_col = detect_column(df, TARGET_COLUMN, ["binary_label", "label", "target", "class"])
    smiles_col = detect_column(df, SMILES_COLUMN, ["smiles"])

    note(
        f"{info['short_name']} verisi okundu",
        f"Satır sayısı: {df.shape[0]}\n"
        f"Kolon sayısı: {df.shape[1]}\n"
        f"Ayraç: {repr(sep)}\n"
        f"Target kolonu: {target_col}\n"
        f"SMILES kolonu: {smiles_col}"
    )
    return df, target_col, smiles_col, info

def clean_target_and_smiles(df, target_col, smiles_col):
    y_numeric = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_numeric.notna() & df[smiles_col].notna()

    df_clean = df.loc[mask].copy().reset_index(drop=True)
    y = pd.to_numeric(df_clean[target_col], errors="coerce").astype(int).to_numpy()

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Binary target bekleniyor. Bulunan sınıflar: {classes}")

    if not set(classes).issubset({0, 1}):
        mapping = {classes[0]: 0, classes[1]: 1}
        y = np.array([mapping[v] for v in y], dtype=int)

    note(
        "Target ve SMILES temizlendi",
        f"Temiz satır sayısı: {len(df_clean)}\n"
        f"Çıkarılan satır sayısı: {len(df) - len(df_clean)}\n"
        f"Sınıf dağılımı: {dict(pd.Series(y).value_counts().sort_index())}"
    )
    return df_clean, y

def load_all_datasets():
    loaded = {}
    for dataset_key in DATASETS:
        df, target_col, smiles_col, info = load_one_dataset(dataset_key)
        df_clean, y = clean_target_and_smiles(df, target_col, smiles_col)
        loaded[dataset_key] = {
            "df": df_clean,
            "y": y,
            "target_col": target_col,
            "smiles_col": smiles_col,
            "info": info
        }
    return loaded

print("Veri okuma fonksiyonları hazır.")

# %% [markdown]
# ## 5. Feature üretim fonksiyonları
# 
# Makine öğrenmesi modeli SMILES metnini doğrudan kullanamaz.  
# Bu nedenle SMILES, sayısal feature matrisine çevrilir.
# 
# Kullanılan temsiller:
# - **Morgan fingerprint:** Atom çevrelerini bit vektörü olarak kodlar.
# - **MACCS fingerprint:** Kimyasal alt yapı anahtarlarını kullanır.
# - **RDKit descriptor:** Molekül ağırlığı, LogP ve benzeri sayısal descriptorlar üretir.

# %%
def _valid_molecules(smiles):
    mols = []
    valid = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        mols.append(mol)
        valid.append(mol is not None)
    return mols, np.array(valid, dtype=bool)

def smiles_to_morgan(smiles):
    mols, valid = _valid_molecules(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    names = [f"Morgan_{i}" for i in range(MORGAN_BITS)]
    rows = []

    for mol, keep in zip(mols, valid):
        if not keep:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)

    return np.vstack(rows), names, valid

def smiles_to_maccs(smiles):
    mols, valid = _valid_molecules(smiles)
    names = [f"MACCS_{i}" for i in range(1, 167)]
    rows = []

    for mol, keep in zip(mols, valid):
        if not keep:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr[1:])

    return np.vstack(rows), names, valid

def smiles_to_rdkit_descriptors(smiles):
    mols, valid = _valid_molecules(smiles)
    descriptor_names = [name for name, _ in Descriptors._descList]
    calc = MolecularDescriptorCalculator(descriptor_names)
    rows = []

    for mol, keep in zip(mols, valid):
        if not keep:
            continue
        desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        rows.append(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0))

    return np.vstack(rows), descriptor_names, valid

def build_features(df, y, smiles_col, feature_set="morgan"):
    smiles = df[smiles_col].tolist()

    if feature_set == "morgan":
        X, names, valid = smiles_to_morgan(smiles)
    elif feature_set == "maccs":
        X, names, valid = smiles_to_maccs(smiles)
    elif feature_set == "rdkit":
        X, names, valid = smiles_to_rdkit_descriptors(smiles)
    elif feature_set == "maccs_morgan":
        X1, n1, v1 = smiles_to_maccs(smiles)
        X2, n2, v2 = smiles_to_morgan(smiles)
        if not np.array_equal(v1, v2):
            raise ValueError("MACCS ve Morgan valid SMILES maskeleri farklı çıktı.")
        X, names, valid = np.hstack([X1, X2]), n1 + n2, v1
    elif feature_set == "morgan_rdkit":
        X1, n1, v1 = smiles_to_morgan(smiles)
        X2, n2, v2 = smiles_to_rdkit_descriptors(smiles)
        if not np.array_equal(v1, v2):
            raise ValueError("Morgan ve RDKit descriptor valid SMILES maskeleri farklı çıktı.")
        X, names, valid = np.hstack([X1, X2]), n1 + n2, v1
    else:
        raise ValueError("feature_set: morgan, maccs, rdkit, maccs_morgan veya morgan_rdkit olmalı.")

    df_valid = df.loc[valid].reset_index(drop=True)
    y_valid = y[valid]

    note(
        f"Feature üretildi: {feature_set}",
        f"Molekül sayısı: {X.shape[0]}\n"
        f"Feature sayısı: {X.shape[1]}\n"
        f"Örnek feature isimleri: {names[:5]}"
    )
    return X, y_valid, names, df_valid

def build_features_for_all(loaded, feature_set="morgan"):
    result = {}
    for dataset_key, data in loaded.items():
        X, y, names, df_valid = build_features(data["df"], data["y"], data["smiles_col"], feature_set)
        result[dataset_key] = {**data, "X": X, "y": y, "feature_names": names, "df": df_valid}
    return result

print("Feature üretim fonksiyonları hazır.")

# %% [markdown]
# ## Metriklerin anlamı
# 
# Bu bölümde model sonuçlarını yorumlamak için kullanılan metrikler özetlenir.
# 
# **Confusion matrix terimleri**
# 
# - **TP / True Positive:** Gerçek sınıf 1, model tahmini 1.
# - **TN / True Negative:** Gerçek sınıf 0, model tahmini 0.
# - **FP / False Positive:** Gerçek sınıf 0, model tahmini 1.
# - **FN / False Negative:** Gerçek sınıf 1, model tahmini 0.
# 
# **Temel formüller**
# 
# - **Recall / Sensitivity = TP / (TP + FN)**  
#   Gerçek pozitiflerin ne kadarının yakalandığını gösterir.
# 
# - **Specificity = TN / (TN + FP)**  
#   Gerçek negatiflerin ne kadarının doğru dışarıda bırakıldığını gösterir.
# 
# - **Precision = TP / (TP + FP)**  
#   Model pozitif dediğinde bunun ne kadar doğru olduğunu gösterir.
# 
# - **F1 = 2 × Precision × Recall / (Precision + Recall)**  
#   Precision ve recall dengesini özetler.
# 
# - **ROC-AUC**  
#   Modelin sınıf 0 ile sınıf 1'i farklı eşik değerlerinde ayırma gücünü özetler.
# 
# - **AP / Average Precision**  
#   Pozitif sınıfı yakalama kalitesini precision-recall mantığıyla özetler.
# 
# - **Balanced Accuracy = (Recall + Specificity) / 2**  
#   Sınıf dağılımı dengesizse normal accuracy değerinden daha dengeli bir özet verir.

# %% [markdown]
# ## 6. Model ve metrik fonksiyonları
# 
# Bu bölüm model eğitimi ve değerlendirme için ortak fonksiyonları içerir.  
# Random Forest ana baseline model olarak kullanılır; çünkü tabular veride güçlü, hızlı ve güvenilir bir başlangıç modelidir.

# %%
def make_rf(n_estimators=300, class_weight="balanced_subsample"):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def get_score_class1(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)

def calculate_metrics(y_true, y_pred, y_score):
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

def train_test_rf(data):
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        data["X"], data["y"], data["df"],
        test_size=TEST_SIZE,
        stratify=data["y"],
        random_state=RANDOM_STATE
    )

    model = make_rf()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = get_score_class1(model, X_test)
    metrics = calculate_metrics(y_test, y_pred, y_score)

    return model, metrics, y_pred, y_score, X_train, X_test, y_train, y_test, df_train, df_test

def save_prediction_table(outdir, name, df_test, smiles_col, y_test, y_pred, y_score):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = pd.DataFrame({
        "SMILES": df_test[smiles_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score_class1": y_score
    })

    path = outdir / f"{name}_predictions.csv"
    pred.to_csv(path, index=False)
    print(f"[Kaydedildi] {path}")
    return pred

print("Model ve metrik fonksiyonları hazır.")

# %% [markdown]
# ## 7. SHAP ve LIME için iki ayrı RF modeli
# 
# Bu bölümde iki veri seti için ayrı RF modeli eğitilir ve açıklama çıktıları üretilir.
# 
# Üretilecek grafikler:
# - SHAP beeswarm
# - SHAP bar
# - SHAP waterfall
# - LIME local explanation
# 
# SHAP waterfall tek bir molekül için tahmini hangi feature'ların yukarı veya aşağı ittiğini gösterir.

# %%
import sys
import subprocess
import importlib.util

if importlib.util.find_spec("shap") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])

if importlib.util.find_spec("lime") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lime"])

import shap
import lime.lime_tabular

lesson_out = ensure_output_root() / "08_interpretability_shap_lime"
lesson_out.mkdir(parents=True, exist_ok=True)

loaded = load_all_datasets()
features = build_features_for_all(loaded, feature_set="maccs")

# %%
def class1_shap_values(explainer, X):
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        values = raw[1]
    else:
        arr = np.asarray(raw)
        values = arr[:, :, 1] if arr.ndim == 3 and arr.shape[-1] == 2 else arr

    ev = np.asarray(explainer.expected_value).ravel()
    base = float(ev[1]) if ev.size > 1 else float(ev[0])
    return values, base

def choose_cases(y_true, y_pred, y_score):
    candidates = []

    def add(label, mask, score, high=True):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        order = np.argsort(score[idx])
        if high:
            order = order[::-1]
        candidates.append((label, int(idx[order[0]])))

    add("correct_positive", (y_true == 1) & (y_pred == 1), y_score, True)
    add("correct_negative", (y_true == 0) & (y_pred == 0), 1 - y_score, True)
    add("false_positive", (y_true == 0) & (y_pred == 1), y_score, True)
    add("false_negative", (y_true == 1) & (y_pred == 0), y_score, False)
    add("borderline", np.ones_like(y_true, dtype=bool), np.abs(y_score - 0.5), False)

    selected = []
    seen = set()
    for label, i in candidates:
        if i not in seen:
            selected.append((label, i))
            seen.add(i)

    return selected

print("SHAP/LIME yardımcı fonksiyonları hazır.")

# %%
metric_rows = []

for dataset_key, data in features.items():
    info = data["info"]
    model_name = f"{info['model_prefix']}_RF_SHAP_LIME"

    note(f"Interpretability modeli eğitiliyor: {model_name}")

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        data["X"], data["y"], data["df"], test_size=TEST_SIZE, stratify=data["y"], random_state=RANDOM_STATE
    )

    rf = make_rf(n_estimators=500)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_score = get_score_class1(rf, X_test)

    metrics = calculate_metrics(y_test, y_pred, y_score)
    metrics.update({"Dataset": dataset_key, "ModelName": model_name})
    metric_rows.append(metrics)

    explainer = shap.TreeExplainer(rf)

    sample_n = min(500, X_test.shape[0])
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(X_test.shape[0], sample_n, replace=False)
    shap_values_sample, base_value = class1_shap_values(explainer, X_test[sample_idx])

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_sample, X_test[sample_idx], feature_names=data["feature_names"], max_display=25, show=False)
    plt.title(f"{model_name} — SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(lesson_out / f"{model_name}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_sample, X_test[sample_idx], feature_names=data["feature_names"], plot_type="bar", max_display=25, show=False)
    plt.title(f"{model_name} — SHAP bar")
    plt.tight_layout()
    plt.savefig(lesson_out / f"{model_name}_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.show()

    shap_values_all, base_value_all = class1_shap_values(explainer, X_test)
    explanation = shap.Explanation(
        values=shap_values_all,
        base_values=np.repeat(base_value_all, X_test.shape[0]),
        data=X_test,
        feature_names=data["feature_names"]
    )

    case_rows = []
    for case_label, i in choose_cases(y_test, y_pred, y_score):
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation[i], max_display=15, show=False)
        plt.title(f"{model_name} — SHAP waterfall — {case_label}")
        plt.tight_layout()

        out_file = lesson_out / f"{model_name}_waterfall_{case_label}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.show()

        case_rows.append({
            "Dataset": dataset_key,
            "case": case_label,
            "index": i,
            "y_true": int(y_test[i]),
            "y_pred": int(y_pred[i]),
            "score": float(y_score[i]),
            "file": str(out_file)
        })

    pd.DataFrame(case_rows).to_csv(lesson_out / f"{model_name}_waterfall_cases.csv", index=False)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=data["feature_names"],
        class_names=["class 0", "class 1"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE
    )

    if len(case_rows) > 0:
        i = int(case_rows[0]["index"])
        lime_exp = lime_explainer.explain_instance(X_test[i], rf.predict_proba, num_features=15, labels=(1,))
        lime_exp.save_to_file(str(lesson_out / f"{model_name}_lime_case.html"))

    print(f"{model_name} SHAP/LIME çıktıları kaydedildi.")

metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(lesson_out / "interpretability_model_metrics.csv", index=False)
display(metric_df)

