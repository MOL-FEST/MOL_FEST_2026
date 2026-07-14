# MOL-FEST 2026 — Makine Öğrenmesi ve Moleküler Veri Analizi Colab Paketi

Bu paket ERα agonist/antagonist sınıflandırma verileri üzerinden moleküler makine öğrenmesi sürecini adım adım göstermek için hazırlanmıştır. Notebooklar Google Colab üzerinde çalışacak şekilde düzenlenmiştir. Her notebookun aynı isimli `.py` versiyonu da vardır.

Ana akış:

```text
veri okuma
→ veri temizleme
→ SMILES'tan fingerprint/descriptor üretme
→ Random Forest baseline
→ 10 model karşılaştırması
→ güçlü aday modellerin seçilmesi
→ random search tuning
→ feature selection
→ feature engineering
→ ensemble / stacking
→ SHAP / LIME açıklanabilirlik
→ over/under sampling
→ baseline RF ile en iyi kombinasyon karşılaştırması
```

## Veri setleri

Notebooklar iki veri setini doğrudan GitHub üzerinden okur:

```text
Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv
Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv
```

Model ön ekleri:

```text
model_ERa_BLA
model_ERa_LUC_VM7
```

## Notebooklar

### 01_data_preparation_and_fingerprints.ipynb

Veriyi okur, target/SMILES kolonlarını kontrol eder, sınıf dağılımını çizer ve MACCS + Morgan fingerprint dosyalarını üretir.

### 02_first_random_forest_model.ipynb

İki veri setinde ilk Random Forest baseline modelini eğitir. Confusion matrix, ROC curve, prediction CSV ve model dosyası üretir.

### 03_train_10_classical_models.ipynb

İki veri setinde 10 klasik modeli karşılaştırır. Tablolar veri setlerine göre ayrıdır. RF'ı geçen güçlü modeller sonraki adımlarda aday model olarak taşınır.

### 04_random_search_tuning.ipynb

RandomForest, ExtraTrees ve HistGradientBoosting için RandomizedSearchCV tuning yapar. Bu notebook model aramasından sonra güçlü adayları iyileştirmek için vardır.

### 05_feature_selection_methods.ipynb

ANOVA, chi-square, mutual information ve RF importance ile top 50/100/150/200 feature seçer. Her seçim RandomForest, ExtraTrees ve HistGradientBoosting ile ayrı ayrı test edilir.

### 06_feature_engineering_and_fingerprint_comparison.ipynb

MACCS, Morgan, RDKit descriptor, MACCS + Morgan, MACCS + RDKit ve Morgan + RDKit feature setlerini karşılaştırır. Feature isimleri ekrana yazdırılır ve CSV olarak kaydedilir.

### 07_pipeline_ensemble_stacking.ipynb

RandomForest, ExtraTrees ve HistGradientBoosting modellerini pipeline, soft voting ve stacking yapıları içinde karşılaştırır. Tablolar veri setlerine göre ayrıdır.

### 08_model_interpretability_shap_lime.ipynb

ExtraTrees modeli üzerinden SHAP beeswarm, SHAP bar, SHAP waterfall ve LIME local explanation çıktıları üretir.

### 09_class_imbalance_resampling.ipynb

RandomForest, ExtraTrees ve HistGradientBoosting için oversampling ve undersampling senaryolarını karşılaştırır. Sampling sadece train set üzerinde yapılır.

### 10_final_reproducible_pipeline.ipynb

Baseline RF ile tüm geliştirilmiş kombinasyonları karşılaştırır. Model ailesi, feature set, feature selection ve sampling kombinasyonlarını test eder. En son baseline RF ile en iyi kombinasyon arasındaki farkı raporlar.

## Temel metrikler

```text
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
Precision = TP / (TP + FP)
F1 = 2 × Precision × Recall / (Precision + Recall)
Balanced Accuracy = (Recall + Specificity) / 2
```

ROC-AUC, modelin sınıfları ayırma gücünü özetler. AP, pozitif sınıfı yakalama kalitesini precision-recall mantığıyla özetler.

## Önerilen anlatım sırası

```text
Ders 1: 01 + 02
Ders 2: 03 + 04
Ders 3: 05 + 06
Ders 4: 07 + 08
Ders 5: 09 + 10
```
