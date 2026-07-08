# Yapay Zeka / Makine Öğrenmesi Workshop Kod Paketi 2026

Bu README, workshop için hazırlanan bütün Python dosyalarının ne yaptığını, neden var olduğunu ve hangi sırayla kullanılabileceğini açıklar.

Ana amaç şudur:

1. CSV verisini okumak.
2. SMILES üzerinden moleküler fingerprint üretmek.
3. Klasik makine öğrenmesi modellerini çalıştırmak.
4. Random Forest üzerinden pipeline, feature selection, resampling ve yorumlanabilirlik analizleri göstermek.
5. LazyPredict, AutoGluon ve AutoKeras gibi automated/AutoML yaklaşımlarına giriş yapmak.
6. SHAP, LIME, PCA, t-SNE, UMAP, RadViz gibi görsel/yorumlanabilir analizleri üretmek.

Varsayılan veri dosyası:

```python
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"
TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"
```

ChEMBL’den hazırlanmış veri kullanacaksan tipik ayar şu olur:

```python
INPUT_FILE = "CHEMBL206_prepared_molecule_classification.csv"
CSV_SEPARATOR = ","
TARGET_COLUMN = "binary_label_active1_inactive0"
SMILES_COLUMN = "QSAR-Ready SMILES"
```

---

# 1. Genel Kurulum

Temel paketler:

```bash
pip install pandas numpy scikit-learn rdkit joblib matplotlib scipy
```

Ek analiz paketleri:

```bash
pip install shap lime umap-learn
```

AutoML paketleri daha ağırdır. Temiz bir `conda` veya `venv` ortamı önerilir:

```bash
pip install lazypredict
pip install autogluon.tabular
pip install autokeras tensorflow
```

---

# 2. Önerilen Workshop Akışı

Yeni başlayanlara uygun kısa akış:

```bash
python 18_feature_extraction_fingerprints.py
python 01_multimodel_classification_benchmark.py
python 02_rf_vs_rf_pipeline_anova.py
python 20_feature_selection_anova_f_classif.py
python 23_feature_selection_model_based_tree_importance.py
python 11_shap_beeswarm_bar_dependence.py
python 12_shap_waterfall_cases.py
```

Daha kapsamlı akış:

```bash
python 31_run_feature_extraction_and_selection_workflow.py
python 14_run_all_individual_analyses.py
python 32_rf_resampling_scenarios.py
```

AutoML bölümünü ayrıca göstermek için:

```bash
python 15_lazypredict_automated_benchmark.py
python 16_autogluon_tabular_automl.py
python 17_autokeras_deep_learning_automl.py
```

---

# 3. Ana Modelleme Scriptleri

## 01_multimodel_classification_benchmark.py

Bu dosya, workshopun temel “bir sürü modeli aynı veride karşılaştıralım” scriptidir.

Ne yapıyor?

- CSV dosyasını okur.
- Hedef kolondaki eksik değerleri temizler.
- SMILES kolonundan Morgan fingerprint üretir.
- Birden fazla klasik makine öğrenmesi modeli çalıştırır.
- Stratified 5-fold cross-validation yapar.
- Her model için metrikleri `mean ± SD` olarak verir.
- Tahminleri ve modelleri kaydeder.

İçindeki model örnekleri:

- Logistic Regression
- kNN
- Linear SVM
- Random Forest
- ExtraTrees
- GaussianNB
- DecisionTree
- Soft Voting
- Stacking

Neden var?

Bu dosya, “aynı veri üzerinde farklı modeller nasıl kıyaslanır?” sorusunu göstermek için var. Workshopta model benchmarking mantığını anlatmak için en iyi başlangıç dosyasıdır.

Tipik çıktı:

```text
multimodel_5cv_results.csv
multimodel_oof_predictions.csv
saved_models_multimodel/
```

---

## 02_rf_vs_rf_pipeline_anova.py

Bu dosya, Random Forest özelinde sade RF ile pipeline içindeki RF’i karşılaştırır.

Ne yapıyor?

- Aynı CSV dosyasını okur.
- SMILES üzerinden fingerprint üretir.
- 5-fold CV yapar.
- Şu tasarımları karşılaştırır:
  - Çıplak RF: scaler yok, feature selection yok.
  - RF + StandardScaler.
  - RF + StandardScaler + ANOVA top 50.
  - RF + StandardScaler + ANOVA top 100.
  - RF + StandardScaler + ANOVA top 150.

Neden var?

Random Forest için önemli öğretici nokta şudur: RF çoğu zaman scaling gerektirmez ve feature selection olmadan da iyi performans verebilir. Bu script bunu deneysel olarak göstermek için var.

Tipik çıktı:

```text
rf_vs_pipeline_anova_5cv_results.csv
rf_vs_pipeline_anova_oof_predictions.csv
rf_pipeline_selected_features_by_fold.csv
saved_models_rf_anova/
```

---

## 03_fetch_prepare_chembl_target_dataset.py

Bu dosya ChEMBL’den veri çekip binary classification dataset’i hazırlar.

Ne yapıyor?

- `TARGET_CHEMBL_ID` değişkenine verilen target için ChEMBL activity verisi çeker.
- IC50, Ki, Kd, EC50 gibi activity tiplerini alır.
- `confidence_score >= 8` filtresi uygular.
- `standard_relation` gibi censored ölçümleri default olarak dışarı atar.
- pChEMBL varsa onu kullanır.
- pChEMBL yoksa nM değerinden pActivity hesaplar:

```python
p_activity = 9 - log10(standard_value_nM)
```

- SMILES temizliği yapar.
- Aynı molekül için tekrar eden ölçümleri median ile birleştirir.
- Binary label üretir.

Neden var?

Workshopta kendi verisi olmayan katılımcılar için ChEMBL’den hızlıca veri çekip modelleme yapılabilmesini sağlar.

Tipik çıktı:

```text
CHEMBL206_raw_chembl_activities.csv
CHEMBL206_cleaned_activity_level.csv
CHEMBL206_prepared_molecule_classification.csv
CHEMBL206_cleaning_report.csv
```

---

# 4. Görsel Analiz ve Yorumlanabilirlik Scriptleri

Bu bölümdeki scriptler model sonuçlarını veya feature uzayını görselleştirmek için var.

## common_data_features.py

Bu dosya doğrudan çalıştırılmaz. Diğer görsel analiz scriptleri tarafından kullanılır.

Ne yapıyor?

- CSV okuma.
- Target kolonu tespit etme.
- SMILES kolonu tespit etme.
- SMILES’tan MACCS veya Morgan fingerprint üretme.
- Random Forest modeli oluşturma.
- SHAP class-1 değerlerini düzgün formatta çekme.
- Ortak yardımcı fonksiyonlar sağlar.

Neden var?

Aynı kodu her analiz scriptine tekrar tekrar yazmamak için var. Kod tekrarını azaltır.

---

## 04_anova_fscore_top30.py

Ne yapıyor?

- Veriyi okur.
- Fingerprint feature’larını üretir veya mevcut feature’ları kullanır.
- ANOVA F-score hesaplar.
- En önemli ilk 30 feature için bar grafiği üretir.

Neden var?

Feature selection kavramına en basit giriş için kullanılır. “Hangi feature hedef sınıfla daha ilişkili?” sorusunu görsel olarak anlatır.

Çıktı:

```text
analysis_outputs/04_anova_fscore/
  anova_f_score_all_features.csv
  anova_f_score_top30.png
```

---

## 05_correlation_heatmap_top30.py

Ne yapıyor?

- ANOVA’ya göre top 30 feature’ı seçer.
- Bu feature’lar ve target arasında Pearson korelasyon matrisi çıkarır.
- Heatmap olarak kaydeder.

Neden var?

Feature’lar arasındaki benzerlik/redundancy ilişkisini göstermek için var. Aynı bilgiyi taşıyan feature’ların fazla olabileceğini anlatır.

Çıktı:

```text
analysis_outputs/05_correlation_heatmap/
  top30_correlation_matrix.csv
  top30_correlation_heatmap.png
```

---

## 06_feature_dendrogram_top30.py

Ne yapıyor?

- ANOVA top 30 feature’ı alır.
- Feature korelasyonuna göre hierarchical clustering yapar.
- Dendrogram çizer.

Neden var?

Benzer feature’ların kümelenmesini göstermek için var. Heatmap’in daha yapısal/kümeli versiyonu gibi düşünülebilir.

Çıktı:

```text
analysis_outputs/06_feature_dendrogram/
  top30_feature_dendrogram.png
```

---

## 07_pca_projection.py

Ne yapıyor?

- Fingerprint feature uzayını PCA ile 2 boyuta indirir.
- Sınıfa göre renklendirilmiş PCA grafiği üretir.
- RF’in class-1 probability skoruna göre renklendirilmiş PCA grafiği üretir.

Neden var?

Yüksek boyutlu moleküler feature uzayının iki boyutta nasıl göründüğünü göstermek için var. PCA lineer bir indirgeme yöntemi olduğu için kolay anlatılır.

Çıktı:

```text
analysis_outputs/07_pca_projection/
  pca_coordinates.csv
  pca_projection_colored_by_class.png
  pca_projection_colored_by_rf_probability.png
```

---

## 08_tsne_projection.py

Ne yapıyor?

- Feature uzayını t-SNE ile 2 boyuta indirir.
- Sınıfa göre ve RF probability’ye göre t-SNE grafikleri üretir.
- Trustworthiness ve normalized Kruskal stress hesaplar.

Neden var?

t-SNE, özellikle yüksek boyutlu veride lokal komşuluk yapısını görselleştirmek için kullanılır. Workshopta PCA ile farkını göstermek için iyi bir örnektir.

Çıktı:

```text
analysis_outputs/08_tsne_projection/
  tsne_coordinates.csv
  tsne_quality_metrics.csv
  tsne_projection_colored_by_class.png
  tsne_projection_colored_by_rf_probability.png
```

---

## 09_umap_projection.py

Ne yapıyor?

- Feature uzayını UMAP ile 2 boyuta indirir.
- Binary fingerprint için Jaccard metriğini, sürekli feature için Euclidean metriğini kullanır.
- Trustworthiness@10 hesaplar.
- Class ve RF probability renkli UMAP grafikleri üretir.

Neden var?

UMAP, t-SNE’ye alternatif modern bir manifold learning yöntemidir. Daha hızlı olabilir ve global yapıyı nispeten daha iyi koruyabilir.

Çıktı:

```text
analysis_outputs/09_umap_projection/
  umap_coordinates.csv
  umap_quality_metrics.csv
  umap_projection_colored_by_class.png
  umap_projection_colored_by_rf_probability.png
```

---

## 10_radviz_top30.py

Ne yapıyor?

- ANOVA top 30 feature’ı seçer.
- RadViz grafiği oluşturur.
- Sınıfların feature uzayında nasıl dağıldığını gösterir.

Neden var?

RadViz, çok boyutlu feature’ları tek bir 2D görsele sıkıştırmak için basit ve görsel olarak etkili bir yöntemdir. Workshopta alternatif visualization örneği olarak iyi çalışır.

Çıktı:

```text
analysis_outputs/10_radviz/
  radviz_top30_anova_features.png
```

---

## 11_shap_beeswarm_bar_dependence.py

Ne yapıyor?

- Random Forest modeli eğitir.
- SHAP değerlerini hesaplar.
- SHAP beeswarm plot üretir.
- SHAP bar plot üretir.
- En önemli birkaç feature için dependence plot üretir.
- Test tahminlerini kaydeder.

Neden var?

Modelin hangi feature’lara dayanarak karar verdiğini açıklamak için var. Global interpretability kısmının ana scriptidir.

Çıktı:

```text
analysis_outputs/11_shap_beeswarm/
  rf_model_used_for_shap.joblib
  shap_values_class1.npy
  shap_top_features.csv
  shap_beeswarm_top25.png
  shap_bar_top25.png
  shap_dependence_*.png
  test_predictions_for_shap.csv
```

---

## 12_shap_waterfall_cases.py

Ne yapıyor?

- Random Forest modeli eğitir.
- SHAP waterfall grafikleri üretir.
- Özellikle şu örnekleri seçer:
  - Doğru ve yüksek güvenli pozitif.
  - Doğru ve yüksek güvenli negatif.
  - False positive varsa.
  - False negative varsa.
  - 0.50’ye yakın borderline tahmin.

Neden var?

Tek bir molekül için modelin kararını adım adım anlatmak için var. Local interpretability kısmında en iyi örneklerden biridir.

Çıktı:

```text
analysis_outputs/12_shap_waterfall/
  rf_model_used_for_waterfall.joblib
  shap_waterfall_cases.csv
  shap_waterfall_*.png
```

---

## 13_lime_local_explanations.py

Ne yapıyor?

- Random Forest modeli eğitir.
- LIME local explanation üretir.
- PNG ve HTML formatında açıklamalar kaydeder.

Neden var?

SHAP’e alternatif bir local explanation yöntemi göstermek için var. “Aynı tahmini farklı explanation yöntemleriyle nasıl açıklarız?” sorusunu anlatır.

Çıktı:

```text
analysis_outputs/13_lime/
  rf_model_used_for_lime.joblib
  lime_sample_*.png
  lime_sample_*.html
  lime_all_local_explanations.csv
```

---

## 14_run_all_individual_analyses.py

Ne yapıyor?

- 04–13 arasındaki görsel analiz scriptlerini sırayla çalıştırır.

Neden var?

Tek tek çalıştırmak yerine tüm görsel analiz paketini otomatik almak için var.

---

# 5. AutoML / Automated ML Scriptleri

## 15_lazypredict_automated_benchmark.py

Ne yapıyor?

- Veriyi okur.
- Fingerprint üretir.
- Train/test split yapar.
- LazyPredict LazyClassifier ile birçok sklearn modelini hızlıca dener.
- Leaderboard kaydeder.
- Mümkünse modelleri kaydeder.
- LazyPredict modeli dışarı vermiyorsa fallback sklearn modelleri eğitir ve kaydeder.
- Tahminleri ve metrikleri kaydeder.

Neden var?

Workshopta “manuel model seçmek yerine otomatik model taraması nasıl yapılır?” sorusunu göstermek için var.

Çıktı:

```text
advanced_outputs/15_lazypredict/
  lazypredict_leaderboard.csv
  lazypredict_raw_prediction_table.csv
  model_metrics_from_saved_models.csv
  all_model_predictions_long.csv
  predictions_*.csv
  saved_models/*.joblib
```

---

## 16_autogluon_tabular_automl.py

Ne yapıyor?

- Veriyi okur.
- Fingerprint feature’larını üretir.
- AutoGluon TabularPredictor ile AutoML modeli eğitir.
- Leaderboard üretir.
- En iyi modeli ve predictor klasörünü kaydeder.
- Test tahminlerini ve metriklerini kaydeder.

Neden var?

AutoGluon, tabular AutoML için güçlü ve pratik bir framework’tür. Workshopta klasik sklearn yaklaşımının üstüne AutoML mantığını göstermek için var.

Çıktı:

```text
advanced_outputs/16_autogluon/
  autogluon_predictor/
  autogluon_leaderboard.csv
  autogluon_test_predictions.csv
  autogluon_test_metrics.csv
  metadata.json
```

---

## 17_autokeras_deep_learning_automl.py

Ne yapıyor?

- Veriyi okur.
- Fingerprint üretir.
- AutoKeras StructuredDataClassifier ile automated deep learning modeli arar.
- En iyi Keras modelini export eder.
- Tahminleri ve metrikleri kaydeder.

Neden var?

Deep learning tarafında “model mimarisini manuel seçmeden otomatik arama yapılabilir mi?” sorusunu göstermek için var.

Çıktı:

```text
advanced_outputs/17_autokeras/
  autokeras_tuner_project/
  autokeras_best_model.keras
  autokeras_test_predictions.csv
  autokeras_test_metrics.csv
  autokeras_feature_metadata.joblib
```

---

# 6. Feature Extraction ve Feature Selection Scriptleri

## common_feature_selection_utils.py

Bu dosya doğrudan çalıştırılmaz. Feature extraction/selection scriptleri için ortak fonksiyonları içerir.

Ne yapıyor?

- CSV okuma.
- Target/SMILES kolonu tespit etme.
- Binary target hazırlama.
- Numeric feature kolonlarını tespit etme.
- Constant feature temizleme.
- Top-k seçilen feature dosyalarını kaydetme.
- Ortak yardımcı fonksiyonlar sağlar.

Neden var?

Feature selection scriptlerinin hepsinde aynı kodu tekrar yazmamak için var.

---

## 18_feature_extraction_fingerprints.py

Ne yapıyor?

- SMILES kolonunu okur.
- RDKit ile molekülleri parse eder.
- Canonical SMILES üretir.
- MACCS fingerprint üretir.
- Morgan fingerprint üretir.
- Avalon fingerprint üretir, eğer RDKit AvalonTools mevcutsa.
- Tüm feature’ları CSV’ye kaydeder.

Neden var?

Modelleme öncesinde molekülleri sayısal feature matrisine çevirmek gerekir. Bu script workshopun feature extraction temel dosyasıdır.

Çıktı:

```text
feature_outputs/18_fingerprints/
  molecules_with_fingerprints.csv
  cleaned_molecules.csv
  feature_manifest.csv
  feature_extraction_report.csv
```

Not:

Avalon her RDKit kurulumunda aktif olmayabilir. Eğer yoksa script MACCS ve Morgan üretmeye devam eder, Avalon’ı skip eder.

---

## 19_feature_selection_variance_threshold.py

Ne yapıyor?

- Feature CSV’sini okur.
- Feature varyanslarını hesaplar.
- En yüksek varyanslı top 50/100/150/200 feature’ı seçer.
- Constant feature’ları ayrıca raporlar.

Neden var?

Feature selection’a en basit giriş yöntemidir. Hedef değişkeni kullanmadığı için unsupervised feature filtering örneğidir.

Çıktı:

```text
feature_outputs/19_variance_threshold/
  variance_threshold_feature_ranking.csv
  variance_threshold_top_50_features.csv
  variance_threshold_top_100_features.csv
  variance_threshold_top_150_features.csv
  variance_threshold_top_200_features.csv
```

---

## 20_feature_selection_anova_f_classif.py

Ne yapıyor?

- ANOVA F-test ile her feature’ın sınıf ayrım gücünü hesaplar.
- Top 50/100/150/200 feature seçer.

Neden var?

Klasik supervised univariate feature selection yöntemini göstermek için var. Hızlıdır ve anlatması kolaydır.

Çıktı:

```text
feature_outputs/20_anova_f_classif/
  anova_f_classif_feature_ranking.csv
  anova_f_classif_top_50_features.csv
  anova_f_classif_top_100_features.csv
  anova_f_classif_top_150_features.csv
  anova_f_classif_top_200_features.csv
```

---

## 21_feature_selection_chi2.py

Ne yapıyor?

- Chi-square testiyle feature seçer.
- Top 50/100/150/200 feature CSV’leri üretir.

Neden var?

Binary fingerprint feature’ları için uygun basit bir supervised seçim yöntemidir. Özellikle 0/1 feature’larda iyi anlatılır.

Çıktı:

```text
feature_outputs/21_chi2/
  chi2_feature_ranking.csv
  chi2_top_50_features.csv
  chi2_top_100_features.csv
  chi2_top_150_features.csv
  chi2_top_200_features.csv
```

---

## 22_feature_selection_mutual_info.py

Ne yapıyor?

- Mutual information ile feature-target ilişkisini ölçer.
- Top 50/100/150/200 feature seçer.

Neden var?

ANOVA lineer/ortalama farkı temelli bir yaklaşımken mutual information daha genel ve nonlineer ilişkileri de yakalayabilir. Bunu göstermek için var.

Çıktı:

```text
feature_outputs/22_mutual_info/
  mutual_info_feature_ranking.csv
  mutual_info_top_50_features.csv
  mutual_info_top_100_features.csv
  mutual_info_top_150_features.csv
  mutual_info_top_200_features.csv
```

---

## 23_feature_selection_model_based_tree_importance.py

Ne yapıyor?

- Random Forest ve ExtraTrees modelleri eğitir.
- Modelin `feature_importances_` değerlerine göre feature ranking yapar.
- Top 50/100/150/200 feature setlerini çıkarır.
- Selector modellerini kaydeder.

Neden var?

Model-based feature selection mantığını göstermek için var. Feature’ın önemini doğrudan bir modelin öğrenilmiş yapısından çıkarır.

Çıktı:

```text
feature_outputs/23_model_based_tree_importance/
  rf_importance_feature_ranking.csv
  rf_importance_top_50_features.csv
  extratrees_importance_feature_ranking.csv
  extratrees_importance_top_50_features.csv
  saved_selector_models/
```

---

## 24_feature_selection_l1_logistic.py

Ne yapıyor?

- L1 penalty kullanan Logistic Regression eğitir.
- Katsayısı sıfıra yaklaşan veya sıfır olan feature’ları eleyebilir.
- Absolute coefficient değerine göre feature ranking yapar.
- Top 50/100/150/200 feature seçer.

Neden var?

Sparse model kavramını anlatmak için var. “Bazı modeller doğal olarak feature selection yapabilir” fikrini gösterir.

Çıktı:

```text
feature_outputs/24_l1_logistic/
  l1_logistic_feature_ranking.csv
  l1_logistic_top_50_features.csv
  l1_logistic_selector.joblib
```

---

## 25_feature_selection_rfe_logistic.py

Ne yapıyor?

- Önce ANOVA ile feature sayısını düşürür.
- Sonra Logistic Regression ile Recursive Feature Elimination yapar.
- Top-k feature setleri üretir.

Neden var?

Wrapper feature selection yöntemini göstermek için var. Univariate yöntemlerden farklı olarak model performansı/estimator yapısı üzerinden iteratif seçim yapar.

Not:

Bu yöntem yavaş olabilir. Workshopta “ileri ama maliyetli yöntem” olarak anlatılabilir.

Çıktı:

```text
feature_outputs/25_rfe_logistic/
  rfe_logistic_feature_ranking.csv
  rfe_logistic_top_50_features.csv
  rfe_logistic_selector.joblib
```

---

## 26_feature_selection_sequential_forward.py

Ne yapıyor?

- Önce ANOVA prefilter yapar.
- Sonra Sequential Forward Selection uygular.
- Default olarak 50 feature seçer.

Neden var?

Feature’ları tek tek ekleyerek model performansını artırmaya çalışan klasik wrapper seçim mantığını göstermek için var.

Not:

Çok yavaş olabilir. Büyük feature setlerinde dikkatli kullanılmalı.

Çıktı:

```text
feature_outputs/26_sequential_forward/
  sequential_forward_feature_ranking.csv
  sequential_forward_top_50_features.csv
  sequential_forward_selector.joblib
```

---

## 27_feature_selection_permutation_importance.py

Ne yapıyor?

- RF modeli eğitir.
- Önce RF impurity importance ile prefilter yapar.
- Sonra held-out test split üzerinde permutation importance hesaplar.
- Top 50/100/150/200 feature seçer.

Neden var?

Permutation importance, feature’ın model performansına gerçek etkisini daha doğrudan ölçmeye çalışır. “Feature’ı karıştırınca model ne kadar bozuluyor?” fikrini gösterir.

Çıktı:

```text
feature_outputs/27_permutation_importance/
  permutation_importance_feature_ranking.csv
  permutation_importance_top_50_features.csv
  permutation_importance_selector_models.joblib
```

---

## 28_dimensionality_reduction_pca.py

Ne yapıyor?

- Feature matrisini PCA ile 50/100/150/200 component’e indirger.
- PCA modellerini kaydeder.
- Explained variance summary üretir.

Neden var?

Bu dosya feature selection değil, dimensionality reduction/feature extraction örneğidir. Workshopta “orijinal feature seçmek” ile “yeni component üretmek” arasındaki farkı anlatmak için var.

Çıktı:

```text
feature_outputs/28_pca_reduction/
  pca_50_components.csv
  pca_100_components.csv
  pca_150_components.csv
  pca_200_components.csv
  pca_model_50.joblib
  pca_explained_variance_summary.csv
```

---

## 29_data_quality_leakage_check.py

Ne yapıyor?

- Veri kalitesi ve leakage kontrolü yapar.
- Target class count raporlar.
- Missing target sayısını raporlar.
- SMILES duplicate kontrolü yapar.
- Invalid SMILES kontrolü yapar.
- Leakage-like kolon isimlerini listeler.
- Constant feature’ları bulur.
- Yüksek korelasyonlu feature çiftlerini raporlar.

Neden var?

Workshopta en kritik mesajlardan biri şudur: Modelden önce veri kontrolü yapılmazsa sonuçlar yanıltıcı olabilir. Bu script bu mesajı göstermek için var.

Çıktı:

```text
feature_outputs/29_data_quality_leakage_check/
  data_quality_report.csv
  suspicious_columns.csv
  duplicate_smiles.csv
  constant_features.csv
  high_correlation_pairs.csv
```

---

## 30_compare_selected_feature_sets.py

Ne yapıyor?

- Farklı feature selection scriptlerinin seçtiği feature setlerini okur.
- Overlap matrix üretir.
- Jaccard similarity matrix üretir.
- Hangi feature’ın kaç yöntem tarafından seçildiğini gösterir.

Neden var?

Farklı feature selection yöntemleri aynı feature’ları mı seçiyor, yoksa farklı biyokimyasal/matematiksel sinyallere mi odaklanıyor? Bu soruyu göstermek için var.

Çıktı:

```text
feature_outputs/30_compare_feature_sets/
  selected_feature_overlap_matrix.csv
  selected_feature_jaccard_matrix.csv
  selected_feature_membership_table.csv
```

---

## 31_run_feature_extraction_and_selection_workflow.py

Ne yapıyor?

- Feature extraction ve feature selection scriptlerini sırayla çalıştırır.
- Default olarak hızlı scriptleri çalıştırır.
- RFE, SFS, permutation importance ve PCA gibi yavaş/opsiyonel scriptleri default olarak skip eder.

Neden var?

Tek komutla temel feature workflow’unu almak için var.

Yavaş scriptleri açmak için:

```python
RUN_SLOW_OPTIONAL = True
```

---

# 7. Resampling Scripti

## 32_rf_resampling_scenarios.py

Ne yapıyor?

- İlk CSV dosyasını okur.
- SMILES üzerinden fingerprint üretir.
- Train/test split yapar.
- Resampling işlemini sadece training set üzerinde yapar.
- Random Forest modeli eğitir.
- Test set üzerinde tahmin yapar.
- Model, tahmin, metrik ve feature importance dosyalarını kaydeder.

Çalıştırdığı senaryolar:

```text
balanced_1_to_1 + oversampling
balanced_1_to_1 + undersampling

positive_5_to_1 + oversampling
positive_5_to_1 + undersampling

negative_5_to_1 + oversampling
negative_5_to_1 + undersampling
```

Neden var?

Class imbalance etkisini göstermek için var. Pozitif ve negatif sınıf oranları değişince RF performansı, recall, specificity, precision gibi metriklerin nasıl değiştiği görülebilir.

Önemli kural:

Test set asla resample edilmez. Sadece train set üzerinde oversampling/undersampling yapılır. Bu leakage’i engeller.

Çıktı:

```text
resampling_outputs/32_rf_resampling/
  rf_resampling_metrics.csv
  all_predictions_long.csv
  train_resampled_class_counts.csv
  train_test_indices.csv
  feature_names.csv
  metadata.json

  saved_models/*.joblib
  predictions/*.csv
  feature_importances/*.csv
```

Ekrana yazdırılan metrikler:

```text
ROC
AP
F1
Accuracy
BalancedAccuracy
Recall/Sensitivity
Specificity
Precision
TN, FP, FN, TP
```

---

# 8. Requirements Dosyaları

## requirements_visual_analyses.txt

Görsel analiz scriptleri için gerekli paketleri listeler.

İçerik:

```text
pandas
numpy
scikit-learn
matplotlib
scipy
rdkit
shap
lime
umap-learn
joblib
```

---

## requirements_advanced_automl.txt

LazyPredict, AutoGluon ve AutoKeras scriptleri için gerekli paketleri listeler.

Not:

AutoGluon ve AutoKeras/TensorFlow ağır paketlerdir. Tek ortamda hepsini kurmak bazen dependency conflict yaratabilir. Gerekirse ayrı ortam kullan.

---

## requirements_feature_workshop.txt

Feature extraction ve feature selection scriptleri için temel paketleri listeler.

İçerik:

```text
pandas
numpy
scikit-learn
scipy
rdkit
joblib
```

---

## requirements_resampling_rf.txt

RF resampling scripti için gereken paketleri listeler.

İçerik:

```text
pandas
numpy
scikit-learn
rdkit
joblib
```

---

# 9. ZIP Dosyaları

## classification_workshop_codes.zip

İlk temel modelleme dosyalarını içerir:

- multimodel benchmark
- RF vs RF pipeline
- ChEMBL data preparation

---

## classification_individual_visual_analysis_scripts.zip

Görsel analiz scriptlerini içerir:

- ANOVA plot
- correlation heatmap
- dendrogram
- PCA
- t-SNE
- UMAP
- RadViz
- SHAP
- LIME

---

## advanced_automated_ml_scripts.zip

Advanced/AutoML scriptlerini içerir:

- LazyPredict
- AutoGluon
- AutoKeras

---

## feature_extraction_selection_workshop_scripts.zip

Feature extraction ve feature selection scriptlerini içerir:

- MACCS/Morgan/Avalon fingerprint
- ANOVA, chi2, mutual information
- RF/ExtraTrees importance
- L1 Logistic
- RFE
- SFS
- permutation importance
- PCA
- leakage/data-quality check
- selected feature set comparison

---

## rf_resampling_scenarios_script.zip

RF resampling senaryosu scriptini içerir:

- balanced
- positive 5x
- negative 5x
- over/under sampling

---

# 10. Workshopta Ek Olarak Anlatılabilecek Güzel Noktalar

## 10.1 Leakage

Model çok iyi çıkıyorsa önce şu sorulmalı:

- Target benzeri kolon feature olarak girmiş mi?
- `class_label`, `class_code`, `activity`, `standard_value`, `pchembl` gibi kolonlar yanlışlıkla feature oldu mu?
- Aynı molekül hem train hem testte var mı?

Bunun için:

```bash
python 29_data_quality_leakage_check.py
```

---

## 10.2 Feature selection her zaman performansı artırmaz

Özellikle Random Forest gibi tree ensemble modeller feature selection olmadan da güçlü olabilir.

Bunu göstermek için:

```bash
python 02_rf_vs_rf_pipeline_anova.py
```

---

## 10.3 Class imbalance metrikleri değiştirir

Accuracy tek başına yeterli değildir. Özellikle imbalance varsa:

- ROC
- AP
- Recall
- Precision
- Specificity
- Balanced Accuracy

birlikte yorumlanmalıdır.

Bunu göstermek için:

```bash
python 32_rf_resampling_scenarios.py
```

---

## 10.4 Global ve local interpretability farklıdır

Global açıklama:

```bash
python 11_shap_beeswarm_bar_dependence.py
```

Local açıklama:

```bash
python 12_shap_waterfall_cases.py
python 13_lime_local_explanations.py
```

---

## 10.5 AutoML sonuç verir ama düşünmeyi kaldırmaz

AutoML modelleri hızlı dener; ama şu konular hâlâ araştırmacının sorumluluğundadır:

- Veri temizliği.
- Leakage kontrolü.
- Doğru split stratejisi.
- Metrik seçimi.
- Modelin biyolojik/kimyasal anlamı.
- External validation.

Bunu göstermek için:

```bash
python 15_lazypredict_automated_benchmark.py
python 16_autogluon_tabular_automl.py
python 17_autokeras_deep_learning_automl.py
```

---

# 11. En Kısa Demo Planı

30 dakikalık kısa workshop için önerilen plan:

1. `18_feature_extraction_fingerprints.py`  
   Molekül nasıl sayıya çevrilir?

2. `01_multimodel_classification_benchmark.py`  
   Birden çok model nasıl karşılaştırılır?

3. `02_rf_vs_rf_pipeline_anova.py`  
   RF neden güçlü baseline’dır?

4. `20_feature_selection_anova_f_classif.py`  
   Feature selection nedir?

5. `23_feature_selection_model_based_tree_importance.py`  
   Model-based feature importance nedir?

6. `11_shap_beeswarm_bar_dependence.py`  
   Global model açıklaması.

7. `12_shap_waterfall_cases.py`  
   Tek molekül için local açıklama.

8. `32_rf_resampling_scenarios.py`  
   Imbalanced data’da sampling etkisi.

---

# 12. Sık Karşılaşılan Hatalar

## RDKit kurulu değil

Hata:

```text
ImportError: RDKit is required
```

Çözüm:

```bash
pip install rdkit
```

veya conda:

```bash
conda install -c conda-forge rdkit
```

---

## Avalon fingerprint çalışmıyor

Bazı RDKit kurulumlarında AvalonTools olmayabilir. Bu durumda script Avalon’ı skip eder; MACCS ve Morgan yine üretilir.

Bu kritik bir hata değildir.

---

## AutoGluon veya AutoKeras kurulurken hata veriyor

Bu paketler ağırdır. Ayrı ortam önerilir:

```bash
conda create -n automl python=3.10
conda activate automl
pip install autogluon.tabular
```

veya:

```bash
conda create -n autokeras python=3.10
conda activate autokeras
pip install autokeras tensorflow
```

---

## Target kolonu bulunamadı

Script içinde şu değişkeni elle düzelt:

```python
TARGET_COLUMN = "senin_target_kolonun"
```

---

## SMILES kolonu bulunamadı

Script içinde şu değişkeni elle düzelt:

```python
SMILES_COLUMN = "senin_smiles_kolonun"
```

---

# 13. Dosya İsimlendirme Mantığı

Dosyalar numaralıdır çünkü workshopta sırayla anlatmak kolay olsun:

```text
01-03  temel modelleme ve ChEMBL veri hazırlama
04-14  görsel analiz ve interpretability
15-17  AutoML / automated ML
18-31  feature extraction ve feature selection
32     resampling senaryoları
```

Bu sıra zorunlu değildir ama anlatım için en pratik sıradır.

---

# 14. Son Not

Bu paket, “tam bilimsel pipeline”dan ziyade workshop/öğretim amaçlı hazırlanmıştır. Yine de temel iyi pratikleri korur:

- Train/test ayrımı yapılır.
- Resampling sadece train set üzerinde yapılır.
- Feature selection çıktıları ayrı kaydedilir.
- Modeller ve tahminler kaydedilir.
- Metrikler CSV olarak dışarı verilir.
- SHAP/LIME gibi yorumlanabilirlik çıktıları üretilir.
- Leakage ve data-quality kontrolü için ayrı script vardır.
