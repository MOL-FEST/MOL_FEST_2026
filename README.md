# MOL-FEST 2026 — Makine Öğrenmesi ve Moleküler Veri Analizi Colab Paketı

Bu paket, ERα agonist/antagonist sınıflandırma verileri üzerinden makine öğrenmesi sürecini adım adım göstermek için hazırlanmıştır. Notebooklar Google Colab üzerinde çalışacak şekilde düzenlenmiştir. Her notebookun aynı isimli `.py` versiyonu da vardır.

Ana akış şu sırayı takip eder:

```text
veri okuma
→ veri temizleme
→ SMILES'tan fingerprint/descriptor üretme
→ Random Forest baseline modeli
→ çoklu model karşılaştırması
→ cross-validation
→ feature selection
→ feature engineering
→ ensemble ve stacking
→ SHAP/LIME ile model açıklama
→ over/under sampling
→ final karşılaştırma tablosu
```

---

## Kullanılan veri setleri

Notebooklar iki veri setini doğrudan GitHub üzerinden okur:

```text
Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv
Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv
```

Notebook içinde bu veri setleri şu isimlerle kullanılır:

```text
ERa_BLA_assay
ERa_LUC_VM7_assay
```

Model çıktılarında karışıklık olmaması için model adları şu ön eklerle kaydedilir:

```text
model_ERa_BLA
model_ERa_LUC_VM7
```

---

## Google Colab'da açma

1. ZIP dosyasını indir.
2. ZIP dosyasını bilgisayarda aç.
3. Google Colab'a gir.
4. `File → Upload notebook` seç.
5. Açmak istediğin `.ipynb` dosyasını yükle.
6. Üst menüden `Runtime → Run all` ile tüm hücreleri çalıştır.

Türkçe arayüzde:

```text
Dosya → Not defteri yükle
Çalışma zamanı → Tümünü çalıştır
Veya
Hücreleri tek tek çalıştır.
```

---

## Çıktı klasörü mimarisi

Tüm notebooklar çıktıları şu klasöre yazar:

```text
molfest_outputs_v3/
```

Bu klasör Colab çalışma alanında oluşur. Notebook çalıştırıldıkça içinde alt klasörler açılır.

Örnek çıktı türleri:

```text
metrics.csv
predictions.csv
.joblib model dosyaları
class distribution grafikleri
confusion matrix grafikleri
ROC curve grafikleri
feature ranking dosyaları
SHAP beeswarm grafikleri
SHAP waterfall grafikleri
LIME HTML açıklamaları
final comparison tabloları
```

---

# Notebook 1 — `01_data_preparation_and_fingerprints.ipynb`

## Ne yapıyor?

Bu notebook iki ERα veri setini okur, temel veri kontrollerini yapar ve molekülleri sayısal feature matrisine dönüştürür.

Yaptığı işlemler:

```text
GitHub CSV dosyalarını okur
target kolonunu bulur
SMILES kolonunu bulur
eksik target/SMILES satırlarını çıkarır
sınıf dağılımı grafiği çizer
MACCS fingerprint üretir
Morgan fingerprint üretir
MACCS + Morgan birleşik feature dosyası kaydeder
```

## Neden var?

Makine öğrenmesi modeli SMILES metnini doğrudan kullanamaz. Moleküllerin önce sayısal vektörlere dönüştürülmesi gerekir. Bu notebook, modelleme öncesindeki temel veri hazırlama ve feature extraction adımını gösterir.

## Ana çıktılar

```text
molfest_outputs_v3/01_data_preparation/dataset_summary.csv
molfest_outputs_v3/01_data_preparation/model_ERa_BLA_class_distribution.png
molfest_outputs_v3/01_data_preparation/model_ERa_LUC_VM7_class_distribution.png
molfest_outputs_v3/01_data_preparation/model_ERa_BLA_maccs_morgan_features.csv
molfest_outputs_v3/01_data_preparation/model_ERa_LUC_VM7_maccs_morgan_features.csv
```

---

# Notebook 2 — `02_first_random_forest_model.ipynb`

## Ne yapıyor?

Bu notebook iki veri seti için ayrı ayrı ilk Random Forest baseline modelini eğitir.

Yaptığı işlemler:

```text
iki veri setini okur
Morgan fingerprint üretir
train/test split yapar
Random Forest modeli eğitir
test set tahminlerini üretir
confusion matrix çizer
ROC curve çizer
model dosyasını kaydeder
tahmin ve metrik tablolarını kaydeder
```

## Neden var?

Random Forest, tabular moleküler verilerde güçlü ve güvenilir bir başlangıç modelidir. Bu notebook, sonraki tüm performans artırma denemeleri için referans alınacak baseline sonucu üretir.

## Ana çıktılar

```text
molfest_outputs_v3/02_first_random_forest/rf_baseline_metrics_both_datasets.csv
molfest_outputs_v3/02_first_random_forest/model_ERa_BLA_rf_baseline_predictions.csv
molfest_outputs_v3/02_first_random_forest/model_ERa_LUC_VM7_rf_baseline_predictions.csv
molfest_outputs_v3/02_first_random_forest/saved_models/model_ERa_BLA_rf_baseline.joblib
molfest_outputs_v3/02_first_random_forest/saved_models/model_ERa_LUC_VM7_rf_baseline.joblib
```

---

# Notebook 3 — `03_train_10_classical_models.ipynb`

## Ne yapıyor?

Bu notebook iki veri seti üzerinde 10 klasik makine öğrenmesi modelini çalıştırır ve karşılaştırır.

Kullanılan modeller:

```text
Logistic Regression
kNN
Linear SVM
RBF SVM
Random Forest
ExtraTrees
Gradient Boosting
HistGradientBoosting
Gaussian Naive Bayes
Decision Tree
```

## Neden var?

Tek bir modelin sonucu yeterli değildir. Aynı veri üzerinde farklı model ailelerini karşılaştırmak, hangi model tipinin veri setine daha uygun olduğunu gösterir. Bu notebook model benchmark mantığını anlatır.

## Ana çıktılar

```text
molfest_outputs_v3/03_train_10_models/ten_model_metrics_both_datasets.csv
molfest_outputs_v3/03_train_10_models/ERa_BLA_assay_model_comparison_roc.png
molfest_outputs_v3/03_train_10_models/ERa_LUC_VM7_assay_model_comparison_roc.png
molfest_outputs_v3/03_train_10_models/saved_models/
```

---

# Notebook 4 — `04_cross_validation_and_model_stability.ipynb`

## Ne yapıyor?

Bu notebook seçilen modelleri 5-fold cross-validation ile değerlendirir.

Yaptığı işlemler:

```text
iki veri setini okur
Morgan fingerprint üretir
5-fold StratifiedKFold tanımlar
modelleri her fold'da eğitir
ortalama ve standart sapma metriklerini hesaplar
ROC-AUC mean ± SD grafiği çizer
```

## Neden var?

Tek train/test split sonucu şansa bağlı olabilir. Cross-validation, modelin farklı veri bölünmelerinde ne kadar stabil olduğunu gösterir. Ortalama skor kadar standart sapma da önemlidir.

## Ana çıktılar

```text
molfest_outputs_v3/04_cross_validation/cv_summary_both_datasets.csv
molfest_outputs_v3/04_cross_validation/cv_fold_metrics_long.csv
molfest_outputs_v3/04_cross_validation/ERa_BLA_assay_cv_roc_mean_sd.png
molfest_outputs_v3/04_cross_validation/ERa_LUC_VM7_assay_cv_roc_mean_sd.png
```

---

# Notebook 5 — `05_feature_selection_methods.ipynb`

## Ne yapıyor?

Bu notebook farklı feature selection yöntemlerini karşılaştırır.

Kullanılan yöntemler:

```text
ANOVA F-test
Chi-square
Mutual Information
Random Forest feature importance
```

Her yöntem için şu feature sayıları denenir:

```text
top 50
top 100
top 150
top 200
```

## Neden var?

Feature sayısı arttıkça model daha fazla bilgi alabilir; fakat gereksiz veya gürültülü feature'lar performansı düşürebilir. Feature selection, daha az ve daha anlamlı feature ile benzer veya daha iyi performans elde etmeyi amaçlar.

## Ana çıktılar

```text
molfest_outputs_v3/05_feature_selection/feature_selection_performance_both_datasets.csv
molfest_outputs_v3/05_feature_selection/model_ERa_BLA_ANOVA_ranking.csv
molfest_outputs_v3/05_feature_selection/model_ERa_LUC_VM7_ANOVA_ranking.csv
molfest_outputs_v3/05_feature_selection/ERa_BLA_assay_feature_selection_roc.png
molfest_outputs_v3/05_feature_selection/ERa_LUC_VM7_assay_feature_selection_roc.png
```

---

# Notebook 6 — `06_feature_engineering_and_fingerprint_comparison.ipynb`

## Ne yapıyor?

Bu notebook farklı moleküler temsil biçimlerini karşılaştırır.

Karşılaştırılan feature setleri:

```text
MACCS
Morgan
RDKit descriptors
MACCS + Morgan
Morgan + RDKit descriptors
```

Her feature seti için aynı Random Forest modeli çalıştırılır.

## Neden var?

Model performansı yalnızca algoritmaya bağlı değildir. Molekülün nasıl temsil edildiği de sonucu ciddi şekilde etkiler. Bu notebook, feature engineering yaklaşımını gösterir.

## Ana çıktılar

```text
molfest_outputs_v3/06_feature_engineering/feature_engineering_comparison_both_datasets.csv
molfest_outputs_v3/06_feature_engineering/ERa_BLA_assay_feature_engineering_roc.png
molfest_outputs_v3/06_feature_engineering/ERa_LUC_VM7_assay_feature_engineering_roc.png
```

---

# Notebook 7 — `07_pipeline_ensemble_stacking.ipynb`

## Ne yapıyor?

Bu notebook pipeline, ensemble ve stacking yapılarını iki veri seti üzerinde dener.

Kullanılan yapılar:

```text
Random Forest pipeline
ExtraTrees pipeline
Soft Voting
Stacking
```

## Neden var?

Pipeline, veri ön işleme ve modeli tek akışta toplar. Ensemble modeller ise birden fazla modelin bilgisini birleştirir. Stacking, ilk seviye modellerin tahminlerini ikinci seviye modele vererek daha kompleks bir karar yapısı kurar.

## Ana çıktılar

```text
molfest_outputs_v3/07_pipeline_ensemble_stacking/pipeline_ensemble_stacking_metrics.csv
molfest_outputs_v3/07_pipeline_ensemble_stacking/ERa_BLA_assay_ensemble_roc.png
molfest_outputs_v3/07_pipeline_ensemble_stacking/ERa_LUC_VM7_assay_ensemble_roc.png
molfest_outputs_v3/07_pipeline_ensemble_stacking/saved_models/
```

---

# Notebook 8 — `08_model_interpretability_shap_lime.ipynb`

## Ne yapıyor?

Bu notebook iki veri seti için ayrı Random Forest modeli eğitir ve model kararlarını açıklamak için SHAP/LIME analizleri üretir.

Üretilen açıklamalar:

```text
SHAP beeswarm
SHAP bar plot
SHAP waterfall
LIME local explanation
```

## Neden var?

Yüksek performans tek başına yeterli değildir. Modelin hangi feature'lara bakarak karar verdiğini anlamak gerekir. SHAP global ve local açıklamalar üretir. LIME ise tek bir örnek etrafında lokal açıklama sağlar.

SHAP waterfall özellikle tek bir molekül için tahminin hangi feature'larla yukarı veya aşağı itildiğini gösterir.

## Ana çıktılar

```text
molfest_outputs_v3/08_interpretability_shap_lime/interpretability_model_metrics.csv
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_BLA_RF_SHAP_LIME_shap_beeswarm.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_BLA_RF_SHAP_LIME_shap_bar.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_BLA_RF_SHAP_LIME_waterfall_*.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_BLA_RF_SHAP_LIME_lime_case.html
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_LUC_VM7_RF_SHAP_LIME_shap_beeswarm.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_LUC_VM7_RF_SHAP_LIME_shap_bar.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_LUC_VM7_RF_SHAP_LIME_waterfall_*.png
molfest_outputs_v3/08_interpretability_shap_lime/model_ERa_LUC_VM7_RF_SHAP_LIME_lime_case.html
```

---

# Notebook 9 — `09_class_imbalance_resampling.ipynb`

## Ne yapıyor?

Bu notebook train set üzerinde farklı sınıf oranları oluşturur ve Random Forest performansını karşılaştırır.

Denenen senaryolar:

```text
pozitif/negatif dengeli
pozitif 5 kat fazla
negatif 5 kat fazla
```

Her senaryo iki sampling tipiyle denenir:

```text
oversampling
undersampling
```

## Neden var?

Sınıf dengesizliği model davranışını değiştirir. Accuracy tek başına yanıltıcı olabilir. Bu notebook sampling stratejilerinin recall, specificity, precision ve balanced accuracy üzerindeki etkisini gösterir.

Test set değiştirilmez. Sampling yalnızca train set üzerinde yapılır.

## Ana çıktılar

```text
molfest_outputs_v3/09_resampling/resampling_metrics_both_datasets.csv
molfest_outputs_v3/09_resampling/model_ERa_BLA_*_confusion_matrix.png
molfest_outputs_v3/09_resampling/model_ERa_LUC_VM7_*_confusion_matrix.png
molfest_outputs_v3/09_resampling/saved_models/
```

---

# Notebook 10 — `10_final_reproducible_pipeline.ipynb`

## Ne yapıyor?

Bu notebook önceki derslerdeki ana stratejileri tek final tabloda karşılaştırır.

Karşılaştırılan stratejiler:

```text
RF baseline
RF + MACCS/Morgan
RF + ANOVA top 100
Voting ensemble
RF + balanced oversampling
```

## Neden var?

Performans artırma tek bir yöntemden ibaret değildir. Feature engineering, feature selection, ensemble ve sampling gibi stratejilerin aynı veri üzerinde kontrollü şekilde karşılaştırılması gerekir. Bu notebook tüm akışı final karşılaştırma tablosuyla özetler.

## Ana çıktılar

```text
molfest_outputs_v3/10_final_pipeline/final_comparison_both_datasets.csv
molfest_outputs_v3/10_final_pipeline/ERa_BLA_assay_final_strategy_roc.png
molfest_outputs_v3/10_final_pipeline/ERa_LUC_VM7_assay_final_strategy_roc.png
molfest_outputs_v3/10_final_pipeline/saved_models/
```

---

# Metriklerin kısa özeti

## Confusion matrix

```text
                 Predicted 0    Predicted 1
Actual 0              TN             FP
Actual 1              FN             TP
```

## Recall

```text
Recall = TP / (TP + FN)
```

Gerçek pozitiflerin ne kadarının yakalandığını gösterir.

## Specificity

```text
Specificity = TN / (TN + FP)
```

Gerçek negatiflerin ne kadarının doğru dışarıda bırakıldığını gösterir.

## Precision

```text
Precision = TP / (TP + FP)
```

Model pozitif dediğinde bunun ne kadar doğru olduğunu gösterir.

## F1

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

Precision ve recall dengesini özetler.

## ROC-AUC

Modelin sınıfları farklı threshold değerlerinde ayırma gücünü özetler.

## AP / Average Precision

Pozitif sınıfı yakalama başarısını precision-recall mantığıyla özetler.

## Balanced Accuracy

```text
Balanced Accuracy = (Recall + Specificity) / 2
```

Sınıf dengesizliği varsa normal accuracy değerinden daha dengeli bir özet verir.

---

# Önerilen anlatım sırası

## Ders 1

```text
01_data_preparation_and_fingerprints.ipynb
02_first_random_forest_model.ipynb
```

## Ders 2

```text
03_train_10_classical_models.ipynb
04_cross_validation_and_model_stability.ipynb
```

## Ders 3

```text
05_feature_selection_methods.ipynb
06_feature_engineering_and_fingerprint_comparison.ipynb
```

## Ders 4

```text
07_pipeline_ensemble_stacking.ipynb
08_model_interpretability_shap_lime.ipynb
```

## Ders 5

```text
09_class_imbalance_resampling.ipynb
10_final_reproducible_pipeline.ipynb
```
