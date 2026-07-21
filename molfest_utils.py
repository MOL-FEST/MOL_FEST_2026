#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MOL-FEST progressive molecular ML pipeline ortak fonksiyonları."""

from pathlib import Path  # Dosya yollarını platformdan bağımsız yönetmek için kullanılır.
import warnings  # Eğitim çıktısını gereksiz uyarılarla kirletmemek için kullanılır.
warnings.filterwarnings("ignore")  # Tekrarlayan sklearn/RDKit uyarılarını gizler.

import numpy as np  # Sayısal matris ve vektör işlemleri için kullanılır.
import pandas as pd  # CSV okuma, tablo birleştirme ve sonuç kaydetme için kullanılır.
import matplotlib.pyplot as plt  # Sınıf dağılımı ve performans grafikleri için kullanılır.
import joblib  # Eğitilmiş sklearn modellerini diske kaydetmek için kullanılır.

from rdkit import Chem, DataStructs  # SMILES parse ve fingerprint array dönüşümü için kullanılır.
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors  # MACCS, Morgan ve descriptor üretimi için kullanılır.
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator  # RDKit descriptor hesaplayıcısıdır.

from sklearn.model_selection import train_test_split  # Stratified train/test split için kullanılır.
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix  # Performans metrikleri için kullanılır.
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier  # Ana modeller ve ensemble yapılarıdır.
from sklearn.pipeline import Pipeline  # Ölçekleme/filtreleme/model adımlarını tek akışa bağlar.
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Ölçekleme gerektiren modeller ve chi2 için kullanılır.
from sklearn.feature_selection import VarianceThreshold, f_classif, chi2, mutual_info_classif  # Feature selection yöntemleri için kullanılır.
from sklearn.linear_model import LogisticRegression  # Lineer model ve stacking final estimator olarak kullanılır.
from sklearn.neighbors import KNeighborsClassifier  # KNN benchmark modeli olarak kullanılır.
from sklearn.svm import SVC, LinearSVC  # SVM benchmark modelleri için kullanılır.
from sklearn.calibration import CalibratedClassifierCV  # LinearSVC'ye predict_proba kazandırmak için kullanılır.
from sklearn.naive_bayes import GaussianNB  # Naive Bayes benchmark modeli olarak kullanılır.
from sklearn.tree import DecisionTreeClassifier  # Tek karar ağacı benchmark modeli olarak kullanılır.

DATASETS = {  # İki veri seti aynı pipeline içinde ayrı ayrı çalıştırılır.
    "ERa_BLA_assay": {  # Birinci veri setinin kısa anahtarıdır.
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv",
        "model_prefix": "model_ERa_BLA",  # Bu veri setindeki model dosyalarının ön ekidir.
        "short_name": "ERα BLA",  # Grafiklerde kullanılacak kısa isimdir.
        "feature_file": "model_ERa_BLA_features.csv",  # Üretilecek hazır feature CSV dosyasıdır.
    },
    "ERa_LUC_VM7_assay": {  # İkinci veri setinin kısa anahtarıdır.
        "url": "https://github.com/MOL-FEST/MOL_FEST_2026/blob/main/Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv",
        "model_prefix": "model_ERa_LUC_VM7",  # Bu veri setindeki model dosyalarının ön ekidir.
        "short_name": "ERα LUC VM7",  # Grafiklerde kullanılacak kısa isimdir.
        "feature_file": "model_ERa_LUC_VM7_features.csv",  # Üretilecek hazır feature CSV dosyasıdır.
    },
}

TARGET_COLUMN = "binary_label_agonist1_antagonist0"  # Binary sınıf etiketini taşıyan kolondur.
SMILES_COLUMN = "QSAR-Ready SMILES"  # Moleküler SMILES bilgisini taşıyan kolondur.
RANDOM_STATE = 42  # Tekrarlanabilir modelleme için kullanılan sabit random seed değeridir.
TEST_SIZE = 0.20  # Test set oranıdır.
MORGAN_BITS = 1024  # Morgan fingerprint uzunluğudur.
MORGAN_RADIUS = 2  # Morgan fingerprint atom çevresi yarıçapıdır.
OUTPUT_ROOT = Path("molfest_outputs")  # Her adımın sonuçları bu klasöre kaydedilir.
FEATURE_STORE = Path("molfest_feature_store")  # Bir kez üretilen feature CSV dosyaları burada tutulur.
FEATURE_BASE_URL = ""  # Feature CSV dosyaları GitHub'a yüklenirse raw klasör URL'si buraya yazılır.

def ensure_dirs():  # Çıktı klasörlerinin varlığını garanti eder.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # Ana sonuç klasörünü oluşturur.
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)  # Feature store klasörünü oluşturur.
    return OUTPUT_ROOT, FEATURE_STORE  # Klasör yollarını geri döndürür.

def note(title, message=""):  # Ekranda düzenli başlık üretmek için kullanılır.
    print("\n" + "=" * 90)  # Görsel ayırıcı çizgi basar.
    print(title)  # Bölüm başlığını yazar.
    print("=" * 90)  # İkinci ayırıcı çizgi basar.
    if message:  # Ek açıklama varsa onu da yazar.
        print(message)

def github_to_raw(url):  # GitHub blob linkini raw CSV linkine dönüştürür.
    return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

def read_csv_flexible(url_or_path):  # CSV ayıracını otomatik bulmaya çalışır.
    source = github_to_raw(str(url_or_path))  # GitHub linki raw formata çevrilir.
    last_error = None  # Son hata mesajı burada tutulur.
    for sep in [";", ","]:  # Önce noktalı virgül, sonra virgül denenir.
        try:  # Okuma denemesi başlatılır.
            df = pd.read_csv(source, sep=sep, encoding="utf-8-sig", low_memory=False)  # CSV dosyası okunur.
            if df.shape[1] > 1:  # Doğru ayraçta birden fazla kolon beklenir.
                return df, sep, source  # Başarılı tablo döndürülür.
        except Exception as exc:  # Hata olursa sonraki ayraç denenir.
            last_error = exc
    raise RuntimeError(f"CSV okunamadı: {url_or_path}\nSon hata: {last_error}")  # İki ayraç da başarısızsa hata verir.

def detect_column(df, preferred, keywords):  # Kolon adını birebir veya anahtar kelime ile bulur.
    if preferred in df.columns:  # Beklenen kolon varsa doğrudan kullanılır.
        return preferred
    for col in df.columns:  # Birebir ad yoksa kolonlar taranır.
        if any(k.lower() in col.lower() for k in keywords):  # Anahtar kelime eşleşmesi aranır.
            return col
    raise ValueError(f"Kolon bulunamadı: {preferred}")  # Kolon bulunamazsa hata verir.

def load_raw_dataset(dataset_key):  # Bir ham veri setini GitHub'dan okur.
    info = DATASETS[dataset_key]  # Veri seti ayarları alınır.
    df, sep, source = read_csv_flexible(info["url"])  # CSV dosyası okunur.
    target_col = detect_column(df, TARGET_COLUMN, ["binary_label", "label", "target", "class"])  # Target kolonu bulunur.
    smiles_col = detect_column(df, SMILES_COLUMN, ["smiles"])  # SMILES kolonu bulunur.
    note(f"{info['short_name']} ham verisi okundu", f"Kaynak: {source}\nSatır: {df.shape[0]}\nKolon: {df.shape[1]}\nAyraç: {repr(sep)}\nTarget: {target_col}\nSMILES: {smiles_col}")
    return df, target_col, smiles_col, info  # Ham veri ve kolon bilgileri döndürülür.

def clean_target_and_smiles(df, target_col, smiles_col):  # Target ve SMILES eksiklerini temizler.
    y_numeric = pd.to_numeric(df[target_col], errors="coerce")  # Target sayısal değere çevrilir.
    mask = y_numeric.notna() & df[smiles_col].notna()  # Target ve SMILES dolu olan satırlar seçilir.
    df_clean = df.loc[mask].copy().reset_index(drop=True)  # Temiz tablo oluşturulur.
    y = pd.to_numeric(df_clean[target_col], errors="coerce").astype(int).to_numpy()  # Target numpy array yapılır.
    classes = np.unique(y)  # Sınıflar kontrol edilir.
    if len(classes) != 2:  # Binary classification beklenir.
        raise ValueError(f"Binary target bekleniyor. Bulunan sınıflar: {classes}")
    if not set(classes).issubset({0, 1}):  # Sınıflar 0/1 değilse map edilir.
        mapping = {classes[0]: 0, classes[1]: 1}  # Küçük değer 0, büyük değer 1 yapılır.
        y = np.array([mapping[v] for v in y], dtype=int)  # Target yeniden oluşturulur.
    note("Target ve SMILES temizlendi", f"Temiz satır: {len(df_clean)}\nÇıkarılan satır: {len(df) - len(df_clean)}\nSınıf dağılımı: {dict(pd.Series(y).value_counts().sort_index())}")
    return df_clean, y  # Temiz tablo ve target döndürülür.

def valid_molecules(smiles):  # SMILES listesinden RDKit molecule nesneleri üretir.
    mols, valid = [], []  # Molekül nesneleri ve geçerlilik bilgileri başlatılır.
    for smi in smiles:  # SMILES satırları tek tek dolaşılır.
        mol = Chem.MolFromSmiles(str(smi))  # SMILES RDKit molecule nesnesine çevrilir.
        mols.append(mol)  # Molekül nesnesi listeye eklenir.
        valid.append(mol is not None)  # Başarılı parse bilgisi tutulur.
    return mols, np.array(valid, dtype=bool)  # Moleküller ve mask döndürülür.

def smiles_to_morgan(smiles):  # Morgan fingerprint üretir.
    mols, valid = valid_molecules(smiles)  # Geçerli moleküller hazırlanır.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)  # Morgan generator tanımlanır.
    names = [f"Morgan_{i}" for i in range(MORGAN_BITS)]  # Morgan feature isimleri üretilir.
    rows = []  # Fingerprint satırları burada tutulur.
    for mol, keep in zip(mols, valid):  # Molekül ve geçerlilik bilgisi dolaşılır.
        if keep:  # Sadece geçerli molekül işlenir.
            fp = generator.GetFingerprint(mol)  # Morgan fingerprint hesaplanır.
            arr = np.zeros((MORGAN_BITS,), dtype=np.float32)  # Boş numpy array açılır.
            DataStructs.ConvertToNumpyArray(fp, arr)  # Fingerprint numpy array'e çevrilir.
            rows.append(arr)  # Satır listeye eklenir.
    return np.vstack(rows), names, valid  # Matris, isimler ve valid mask döndürülür.

def smiles_to_maccs(smiles):  # MACCS fingerprint üretir.
    mols, valid = valid_molecules(smiles)  # Geçerli moleküller hazırlanır.
    names = [f"MACCS_{i}" for i in range(1, 167)]  # MACCS feature isimleri üretilir.
    rows = []  # MACCS satırları burada tutulur.
    for mol, keep in zip(mols, valid):  # Molekül ve geçerlilik dolaşılır.
        if keep:  # Sadece geçerli molekül işlenir.
            fp = MACCSkeys.GenMACCSKeys(mol)  # MACCS key hesaplanır.
            arr = np.zeros((167,), dtype=np.float32)  # RDKit MACCS array alanı açılır.
            DataStructs.ConvertToNumpyArray(fp, arr)  # RDKit fingerprint numpy array'e çevrilir.
            rows.append(arr[1:])  # 0. bit çıkarılır.
    return np.vstack(rows), names, valid  # Matris, isimler ve valid mask döndürülür.

def smiles_to_rdkit_descriptors(smiles):  # RDKit descriptor üretir.
    mols, valid = valid_molecules(smiles)  # Geçerli moleküller hazırlanır.
    descriptor_names = [name for name, _ in Descriptors._descList]  # RDKit descriptor isimleri alınır.
    calc = MolecularDescriptorCalculator(descriptor_names)  # Descriptor hesaplayıcı oluşturulur.
    rows = []  # Descriptor satırları burada tutulur.
    for mol, keep in zip(mols, valid):  # Molekül ve geçerlilik dolaşılır.
        if keep:  # Sadece geçerli molekül işlenir.
            desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)  # Descriptor değerleri hesaplanır.
            rows.append(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0))  # NaN ve inf değerler temizlenir.
    names = [f"RDKit_{name}" for name in descriptor_names]  # Descriptor isimlerine RDKit prefix eklenir.
    return np.vstack(rows), names, valid  # Matris, isimler ve valid mask döndürülür.

def smiles_to_avalon(smiles):  # Avalon fingerprint üretmeye çalışır.
    try:  # Avalon modülü her RDKit kurulumunda bulunmayabilir.
        from rdkit.Avalon import pyAvalonTools  # Avalon araçları çağrılır.
    except Exception:  # Avalon yoksa güvenli şekilde atlanır.
        return None, [], None
    mols, valid = valid_molecules(smiles)  # Geçerli moleküller hazırlanır.
    names = [f"Avalon_{i}" for i in range(1024)]  # Avalon feature isimleri üretilir.
    rows = []  # Avalon satırları burada tutulur.
    for mol, keep in zip(mols, valid):  # Molekül ve geçerlilik dolaşılır.
        if keep:  # Sadece geçerli molekül işlenir.
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024)  # Avalon fingerprint hesaplanır.
            arr = np.zeros((1024,), dtype=np.float32)  # Boş numpy array açılır.
            DataStructs.ConvertToNumpyArray(fp, arr)  # Fingerprint numpy array'e çevrilir.
            rows.append(arr)  # Satır listeye eklenir.
    return np.vstack(rows), names, valid  # Matris, isimler ve valid mask döndürülür.

def generate_feature_table(dataset_key):  # Tek veri seti için tüm feature tablosunu üretir.
    df, target_col, smiles_col, info = load_raw_dataset(dataset_key)  # Ham veri okunur.
    df_clean, y = clean_target_and_smiles(df, target_col, smiles_col)  # Target/SMILES temizlenir.
    smiles = df_clean[smiles_col].tolist()  # SMILES listesi alınır.
    X_morgan, n_morgan, valid_morgan = smiles_to_morgan(smiles)  # Morgan feature üretilir.
    X_maccs, n_maccs, valid_maccs = smiles_to_maccs(smiles)  # MACCS feature üretilir.
    X_rdkit, n_rdkit, valid_rdkit = smiles_to_rdkit_descriptors(smiles)  # RDKit descriptor üretilir.
    X_avalon, n_avalon, valid_avalon = smiles_to_avalon(smiles)  # Avalon varsa üretilir.
    valid = valid_morgan & valid_maccs & valid_rdkit  # Ortak geçerli molekül maskesi belirlenir.
    if valid_avalon is not None:  # Avalon varsa ortak maskeye eklenir.
        valid = valid & valid_avalon
    df_valid = df_clean.loc[valid].reset_index(drop=True)  # Geçerli moleküller tabloya alınır.
    y_valid = y[valid]  # Target geçerli moleküllere göre filtrelenir.
    blocks = [X_morgan, X_maccs, X_rdkit]  # Temel feature blokları listelenir.
    names = n_morgan + n_maccs + n_rdkit  # Temel feature isimleri birleştirilir.
    if X_avalon is not None:  # Avalon başarılıysa eklenir.
        blocks.append(X_avalon)  # Avalon matrisi eklenir.
        names += n_avalon  # Avalon isimleri eklenir.
    X_all = np.hstack(blocks)  # Tüm feature blokları birleştirilir.
    feature_df = pd.concat([pd.DataFrame({"Dataset": dataset_key, "SMILES": df_valid[smiles_col].values, "Target": y_valid}), pd.DataFrame(X_all, columns=names)], axis=1)  # Metadata ve featurelar birleştirilir.
    return feature_df, names, info  # Feature tablosu, feature isimleri ve info döndürülür.

def generate_feature_store():  # İki veri seti için feature store oluşturur.
    ensure_dirs()  # Klasörler oluşturulur.
    index_rows = []  # Feature store özeti burada tutulur.
    for dataset_key, info in DATASETS.items():  # Her veri seti dolaşılır.
        feature_df, feature_names, info = generate_feature_table(dataset_key)  # Feature tablosu üretilir.
        out_file = FEATURE_STORE / info["feature_file"]  # Feature CSV yolu belirlenir.
        feature_df.to_csv(out_file, index=False)  # Feature tablosu kaydedilir.
        manifest_file = FEATURE_STORE / f"{info['model_prefix']}_feature_manifest.csv"  # Manifest yolu belirlenir.
        pd.DataFrame({"Feature": feature_names}).to_csv(manifest_file, index=False)  # Feature isimleri kaydedilir.
        has_avalon = any(name.startswith("Avalon_") for name in feature_names)  # Avalon varlığı kontrol edilir.
        index_rows.append({"Dataset": dataset_key, "FeatureFile": str(out_file), "ManifestFile": str(manifest_file), "Rows": feature_df.shape[0], "FeatureCount": len(feature_names), "HasAvalon": has_avalon})  # Özet satırı eklenir.
    index_df = pd.DataFrame(index_rows)  # Özet tablo oluşturulur.
    index_df.to_csv(FEATURE_STORE / "feature_store_index.csv", index=False)  # Özet tablo kaydedilir.
    return index_df  # Özet tablo döndürülür.

def feature_file_path(dataset_key):  # Feature CSV için local veya GitHub yolunu döndürür.
    info = DATASETS[dataset_key]  # Veri seti bilgileri alınır.
    local_path = FEATURE_STORE / info["feature_file"]  # Local feature dosyası yolu oluşturulur.
    if local_path.exists():  # Local dosya varsa onu kullanır.
        return local_path
    if FEATURE_BASE_URL:  # Local dosya yoksa GitHub raw base URL kullanılır.
        return f"{FEATURE_BASE_URL.rstrip('/')}/{info['feature_file']}"
    raise FileNotFoundError(f"Feature dosyası bulunamadı: {local_path}. Önce 01_generate_feature_store çalıştırılmalı.")  # Dosya bulunamazsa hata verilir.

def read_feature_table(dataset_key):  # Hazır feature CSV dosyasını okur.
    path = feature_file_path(dataset_key)  # Dosya yolu veya URL alınır.
    df = pd.read_csv(path)  # Feature tablosu okunur.
    note(DATASETS[dataset_key]["short_name"] + " feature tablosu okundu", f"Kaynak: {path}\nSatır: {df.shape[0]}\nKolon: {df.shape[1]}")
    return df  # Feature tablosu döndürülür.

def feature_columns(df, feature_set):  # İstenen feature setinin kolonlarını döndürür.
    groups = {"morgan": ["Morgan_"], "maccs": ["MACCS_"], "rdkit": ["RDKit_"], "avalon": ["Avalon_"], "maccs_morgan": ["MACCS_", "Morgan_"], "maccs_rdkit": ["MACCS_", "RDKit_"], "morgan_rdkit": ["Morgan_", "RDKit_"], "all_available": ["Morgan_", "MACCS_", "RDKit_", "Avalon_"]}  # Prefix haritası tanımlanır.
    prefixes = groups[feature_set]  # İlgili prefixler alınır.
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]  # Prefixlere uyan kolonlar seçilir.
    if not cols:  # Hiç kolon yoksa hata verilir.
        raise ValueError(f"{feature_set} için feature kolonu bulunamadı.")
    return cols  # Feature kolonları döndürülür.

def available_feature_sets(df):  # Mevcut feature setlerini listeler.
    sets = ["morgan", "maccs", "rdkit", "maccs_morgan", "maccs_rdkit", "morgan_rdkit", "all_available"]  # Temel setler tanımlanır.
    if any(c.startswith("Avalon_") for c in df.columns):  # Avalon kolonu varsa listeye eklenir.
        sets.insert(3, "avalon")
    return sets  # Set listesi döndürülür.

def split_xy(df, cols):  # Feature tablosundan stratified train/test split üretir.
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)  # Feature matrisi temizlenir.
    y = df["Target"].astype(int).to_numpy()  # Target array oluşturulur.
    return train_test_split(X, y, df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)  # Split döndürülür.

def get_score_class1(model, X):  # Modelden sınıf 1 skoru alır.
    if hasattr(model, "predict_proba"):  # Model olasılık veriyorsa kullanılır.
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):  # Model decision score veriyorsa kullanılır.
        return model.decision_function(X)
    return model.predict(X).astype(float)  # Son seçenek olarak tahmin sınıfı skor gibi kullanılır.

def calculate_metrics(y_true, y_pred, y_score):  # Performans metriklerini hesaplar.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()  # Confusion matrix değerleri alınır.
    specificity = tn / (tn + fp) if (tn + fp) else np.nan  # Specificity hesaplanır.
    return {"ROC": roc_auc_score(y_true, y_score), "AP": average_precision_score(y_true, y_score), "F1": f1_score(y_true, y_pred, zero_division=0), "Accuracy": accuracy_score(y_true, y_pred), "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred), "Recall": recall_score(y_true, y_pred, zero_division=0), "Specificity": specificity, "Precision": precision_score(y_true, y_pred, zero_division=0), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}  # Metrikler döndürülür.

def make_model(model_type, random_state=RANDOM_STATE):  # Model tipine göre sklearn modeli üretir.
    if model_type == "RandomForest": return RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)  # Random Forest döndürülür.
    if model_type == "ExtraTrees": return ExtraTreesClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=random_state, n_jobs=-1)  # ExtraTrees döndürülür.
    if model_type == "HistGradientBoosting": return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=random_state)  # HistGradientBoosting döndürülür.
    if model_type == "LogisticRegression": return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))])  # Logistic Regression pipeline döndürülür.
    if model_type == "kNN": return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))])  # kNN pipeline döndürülür.
    if model_type == "LinearSVM": return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=random_state), cv=3))])  # Linear SVM pipeline döndürülür.
    if model_type == "RBF_SVM": return Pipeline([("variance", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=random_state))])  # RBF SVM pipeline döndürülür.
    if model_type == "GradientBoosting": return GradientBoostingClassifier(random_state=random_state)  # GradientBoosting döndürülür.
    if model_type == "GaussianNB": return GaussianNB()  # GaussianNB döndürülür.
    if model_type == "DecisionTree": return DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=random_state)  # DecisionTree döndürülür.
    raise ValueError(f"Bilinmeyen model tipi: {model_type}")  # Bilinmeyen model tipi hata verir.

def candidate_model_types():  # Pipeline boyunca taşınan ana aday modellerdir.
    return ["RandomForest", "ExtraTrees", "HistGradientBoosting"]

def ten_model_types():  # 10 model aramasında kullanılacak model listesidir.
    return ["LogisticRegression", "kNN", "LinearSVM", "RBF_SVM", "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "GaussianNB", "DecisionTree"]

def fit_evaluate(model, X_train, X_test, y_train, y_test):  # Modeli eğitir ve test skorlarını hesaplar.
    model.fit(X_train, y_train)  # Model train set üzerinde eğitilir.
    y_pred = model.predict(X_test)  # Test sınıf tahminleri alınır.
    y_score = get_score_class1(model, X_test)  # Test sınıf 1 skorları alınır.
    metrics = calculate_metrics(y_test, y_pred, y_score)  # Metrikler hesaplanır.
    return model, y_pred, y_score, metrics  # Sonuçlar döndürülür.

def save_model(model, feature_names, path, extra=None):  # Modeli feature listesiyle birlikte kaydeder.
    payload = {"model": model, "feature_names": feature_names}  # Model paketi oluşturulur.
    if extra: payload.update(extra)  # Ek metadata varsa pakete eklenir.
    path = Path(path)  # Yol Path nesnesine çevrilir.
    path.parent.mkdir(parents=True, exist_ok=True)  # Klasör oluşturulur.
    joblib.dump(payload, path)  # Model joblib olarak kaydedilir.

def save_predictions(df_test, y_test, y_pred, y_score, path):  # Test tahminlerini CSV olarak kaydeder.
    pred = pd.DataFrame({"SMILES": df_test["SMILES"].values, "y_true": y_test, "y_pred": y_pred, "y_score_class1": y_score})  # Tahmin tablosu oluşturulur.
    path = Path(path)  # Yol Path nesnesine çevrilir.
    path.parent.mkdir(parents=True, exist_ok=True)  # Klasör oluşturulur.
    pred.to_csv(path, index=False)  # CSV kaydedilir.
    return pred  # Tahmin tablosu döndürülür.

def select_feature_ranking(method, X_train, y_train, feature_names):  # Feature selection sıralaması üretir.
    if method == "none": return list(range(len(feature_names)))  # Seçim yoksa mevcut sıra döner.
    if method == "ANOVA": scores, _ = f_classif(X_train, y_train); return np.argsort(np.nan_to_num(scores))[::-1].tolist()  # ANOVA skoru ile sıralanır.
    if method == "Chi2": X_chi = MinMaxScaler().fit_transform(X_train) if np.nanmin(X_train) < 0 else X_train; scores, _ = chi2(X_chi, y_train); return np.argsort(np.nan_to_num(scores))[::-1].tolist()  # Chi2 skoru ile sıralanır.
    if method == "MutualInfo": scores = mutual_info_classif(X_train, y_train, discrete_features="auto", random_state=RANDOM_STATE); return np.argsort(np.nan_to_num(scores))[::-1].tolist()  # Mutual information ile sıralanır.
    if method == "RF_importance": model = make_model("RandomForest"); model.fit(X_train, y_train); return np.argsort(model.feature_importances_)[::-1].tolist()  # RF importance ile sıralanır.
    raise ValueError(f"Bilinmeyen feature selection yöntemi: {method}")  # Bilinmeyen yöntem hata verir.

def write_feature_list(features, path):  # Feature isimlerini txt dosyasına yazar.
    path = Path(path)  # Yol Path nesnesine çevrilir.
    path.parent.mkdir(parents=True, exist_ok=True)  # Klasör oluşturulur.
    path.write_text("\n".join(features), encoding="utf-8")  # Feature isimleri satır satır yazılır.

def read_feature_list(path):  # Txt dosyasından feature isimlerini okur.
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]  # Boş satırlar çıkarılır.

def apply_resampling(y_train, ratio=1.0, method="none"):  # Sampling sonrası train indexlerini üretir.
    if method == "none": return np.arange(len(y_train))  # Sampling yoksa tüm indexler döner.
    rng = np.random.RandomState(RANDOM_STATE)  # Sabit random generator oluşturulur.
    pos = np.where(y_train == 1)[0]  # Pozitif sınıf indexleri alınır.
    neg = np.where(y_train == 0)[0]  # Negatif sınıf indexleri alınır.
    current = len(pos) / len(neg)  # Mevcut pozitif/negatif oranı hesaplanır.
    if method == "oversampling":  # Oversampling seçilmişse örnek çoğaltılır.
        n_pos, n_neg = (int(np.ceil(ratio * len(neg))), len(neg)) if current < ratio else (len(pos), int(np.ceil(len(pos) / ratio)))  # Hedef sınıf sayıları hesaplanır.
        s_pos = rng.choice(pos, n_pos, replace=n_pos > len(pos))  # Pozitif örnekler seçilir.
        s_neg = rng.choice(neg, n_neg, replace=n_neg > len(neg))  # Negatif örnekler seçilir.
    elif method == "undersampling":  # Undersampling seçilmişse örnek azaltılır.
        n_pos, n_neg = (len(pos), min(len(neg), max(5, int(np.floor(len(pos) / ratio))))) if current < ratio else (min(len(pos), max(5, int(np.floor(ratio * len(neg))))), len(neg))  # Hedef sınıf sayıları hesaplanır.
        s_pos = rng.choice(pos, n_pos, replace=False)  # Pozitif örnekler seçilir.
        s_neg = rng.choice(neg, n_neg, replace=False)  # Negatif örnekler seçilir.
    else:  # Bilinmeyen sampling yöntemi hata verir.
        raise ValueError("method none, oversampling veya undersampling olmalı.")
    idx = np.concatenate([s_pos, s_neg])  # Pozitif ve negatif indexler birleştirilir.
    rng.shuffle(idx)  # Index sırası karıştırılır.
    return idx  # Sampling sonrası indexler döndürülür.
