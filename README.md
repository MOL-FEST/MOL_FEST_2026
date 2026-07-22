# MOL-FEST 2026 — Makine Öğrenmesi ve Moleküler Veri Analizi Colab Paketi

Bu paket, ERα agonist/antagonist sınıflandırma verileri üzerinden moleküler makine öğrenmesi sürecini adım adım göstermek için hazırlanmıştır. Notebooklar Google Colab üzerinde çalışacak şekilde düzenlenmiştir.

Final raporlama ve aday toplama işlemi `09_collect_best_candidates.ipynb` içinde yapılır.

---

## 1. Genel amaç

Bu ders paketi, moleküler sınıflandırma problemlerinde performansı adım adım artırma mantığını göstermek için hazırlanmıştır.

Ana fikir şudur:

```text
Basit ve güvenilir bir Random Forest + Morgan fingerprint baseline kur
→ farklı feature setlerini dene
→ feature selection ile iyileşme var mı kontrol et
→ klasik modelleri karşılaştır
→ sampling ile sınıf dengesizliğini yönet
→ RF tuning yap
→ voting / stacking / multiview gibi advanced yapıları dene
→ tüm adayları tek yerde topla ve baseline ile karşılaştır
```

Bu nedenle paket yalnızca model eğitmek için değil, aynı zamanda bir makine öğrenmesi pipeline'ının nasıl karar vere vere geliştirildiğini göstermek için tasarlanmıştır.

---

## 2. Veri setleri

Notebooklar iki ERα agonist/antagonist veri setiyle çalışır:

```text
Train_2_1_A14_A17_ERa_BLA_agonist_antagonist.csv
Train_2_2_A15_A18_ERa_LUC_VM7_agonist_antagonist.csv
```

Kullanılan iki pipeline adı:

```text
ERa_BLA_assay
ERa_LUC_VM7_assay
```

Model ön ekleri:

```text
model_ERa_BLA
model_ERa_LUC_VM7
```

Target kolonu:

```text
binary_label_agonist1_antagonist0
```

SMILES kolonu:

```text
QSAR-Ready SMILES
```

---

## 3. GitHub ve çıktı klasör yapısı

Notebooklar ham veri ve hazır sonuç dosyalarını GitHub üzerinden okuyacak şekilde düzenlenmiştir.

Önerilen repo yapısı:

```text
MOL_FEST_2026/
├── README.md
├── 01_generate_feature_store.ipynb
├── 02_rf_morgan_baseline.ipynb
├── 03_rf_feature_ablation.ipynb
├── 04_feature_selection.ipynb
├── 05_train_12_models_to_search_a_model.ipynb
├── 06_resampling_search.ipynb
├── 07_random_search_tuning_test_performance_candidates_fixed.ipynb
├── 08_advanced_architectures.ipynb
├── 09_collect_best_candidates.ipynb
└── molfest_outputs/
    ├── 01_feature_store/
    ├── 03_rf_feature_ablation/
    ├── 04_feature_selection/
    ├── 05_train_12_models/
    ├── 06_resampling/
    ├── 07_random_search_tuning_fast/
    ├── 08_advanced_ensembles/
    └── 09_collect_best_candidates/
```

`molfest_outputs/` ana çıktı klasörüdür. Feature CSV dosyaları da artık ayrı `molfest_feature_store/` klasöründe değil, şu klasörde tutulur:

```text
molfest_outputs/01_feature_store/
```

Bu klasörde beklenen temel dosyalar:

```text
model_ERa_BLA_features.csv
model_ERa_LUC_VM7_features.csv
feature_store_index.csv
```

---

## 4. Notebook akışı

### 01_generate_feature_store.ipynb

Ham CSV dosyalarını okur, target ve SMILES kolonlarını kontrol eder, eksik target/SMILES satırlarını temizler ve moleküler feature dosyalarını üretir.

Üretilen feature grupları:

```text
Morgan fingerprint
MACCS keys
Avalon fingerprint
RDKit descriptors
```

Bu notebookun temel amacı, sonraki adımlarda tekrar tekrar fingerprint üretmemek için hazır feature CSV dosyaları oluşturmaktır.

Çıktı klasörü:

```text
molfest_outputs/01_feature_store/
```

Ana çıktılar:

```text
model_ERa_BLA_features.csv
model_ERa_LUC_VM7_features.csv
feature_store_index.csv
```

---

### 02_rf_morgan_baseline.ipynb

İlk referans modeli kurar.

Model:

```text
RandomForestClassifier
```

Feature set:

```text
Morgan fingerprint
```

Bu adım, sonraki bütün iyileştirmeler için başlangıç noktasıdır. Yani bu notebooktaki `RF + Morgan + no sampling` sonucu ana baseline olarak kullanılır.

Üretilen çıktılar:

```text
baseline metrik tablosu
prediction CSV dosyaları
ROC grafikleri
confusion matrix grafikleri
eğitilmiş model dosyaları
```

---

### 03_rf_feature_ablation.ipynb

Random Forest sabit tutularak farklı feature setleri karşılaştırılır.

Deney mantığı:

```text
RF + Morgan
RF + MACCS
RF + Avalon
RF + RDKit
RF + MACCS + Morgan
RF + MACCS + RDKit
RF + Morgan + RDKit
RF + Avalon + RDKit
RF + all_available
```

Amaç, model değiştirmeden yalnızca feature temsilinin performansa etkisini görmektir.

Bu adımda iki veri seti için en iyi feature stratejileri ayrı ayrı belirlenir. Mevcut akışta kullanılan pratik karar:

```text
ERa_BLA_assay      → rdkit
ERa_LUC_VM7_assay  → all_available
```

Çıktı klasörü:

```text
molfest_outputs/03_rf_feature_ablation/
```

---

### 04_feature_selection.ipynb

Step 03 ile seçilen feature setleri üzerinde feature selection yöntemleri denenir.

Kullanılan yöntemler:

```text
ANOVA F-score
Chi-square
Mutual information
Random Forest feature importance
```

ERa_BLA için RDKit feature sayısı daha düşük olduğu için daha küçük top-k değerleri denenir:

```text
50, 100, 150, 200
```

ERa_LUC_VM7 için tüm feature seti daha büyük olduğu için daha geniş top-k değerleri denenir:

```text
250, 750, 1250, 1750
```

Amaç, feature sayısını azaltmanın ROC-AUC, AP ve F1 üzerinde anlamlı bir iyileşme sağlayıp sağlamadığını test etmektir. Eğer iyileşme anlamlı değilse, sonraki adımda Step 03'te seçilen feature setleriyle devam edilir.

Çıktı klasörü:

```text
molfest_outputs/04_feature_selection/
```

---

### 05_train_12_models_to_search_a_model.ipynb

Seçilen gatekeeper feature setleri üzerinde farklı klasik makine öğrenmesi modelleri denenir.

Temel modeller:

```text
Logistic Regression
kNN
Linear SVM
RBF SVM
Random Forest
Extra Trees
Gradient Boosting
HistGradientBoosting
Gaussian Naive Bayes
Decision Tree
XGBoost
CatBoost
```

XGBoost ve CatBoost ortamda kurulu değilse notebook bunları kurmaya çalışır veya uygun şekilde atlar.

Bu adımın amacı, Random Forest'ın iyi bir gatekeeper olup olmadığını görmek ve diğer güçlü modellerle karşılaştırmaktır. Eğer farklı modeller çok güçlü görünse bile, bu pipeline'da yorumlanabilirlik, stabilite ve eğitim akışı açısından Random Forest ile devam edilebilir.

Çıktı klasörü:

```text
molfest_outputs/05_train_12_models/
```

---

### 06_resampling_search.ipynb

Random Forest sabit tutularak sınıf dengesizliği senaryoları test edilir.

Önemli nokta:

```text
Sampling sadece train set üzerinde yapılır.
Test set doğal sınıf dağılımında bırakılır.
```

Denenen senaryolar:

```text
none
balanced_oversampling
balanced_undersampling
positive_5x_oversampling
negative_5x_oversampling
positive_5x_undersampling
negative_5x_undersampling
```

Amaç, sınıf dağılımına müdahalenin ROC-AUC, AP, F1, recall ve specificity üzerindeki etkisini görmektir.

Bu adım sonunda her veri seti için en iyi RF + sampling kombinasyonu seçilir.

Çıktı klasörü:

```text
molfest_outputs/06_resampling/
```

---

### 07_random_search_tuning_test_performance_candidates_fixed.ipynb

Step 06 ile seçilen RF + sampling gatekeeper üzerinde Random Search tuning yapılır.

Bu notebookta random search adayları yalnızca internal CV açısından değil, ayrıca doğal test set üzerinde de raporlanır.

Raporlanan aday performansları:

```text
Test_ROC
Test_AP
Test_F1
Test_BalancedAccuracy
Test_Recall
Test_Specificity
Test_Precision
TN / FP / FN / TP
```

Amaç, RF hiperparametre değişiminin gerçekten test set performansını artırıp artırmadığını göstermektir.

Bu adımda aday tabloları iki pipeline için ayrı ayrı verilir:

```text
ERa_BLA_assay
ERa_LUC_VM7_assay
```

Çıktı klasörü:

```text
molfest_outputs/07_random_search_tuning_fast/
```

---

### 08_advanced_architectures.ipynb

Bu notebook advanced model mimarilerini test eder.

Denenen yapılar:

```text
Voting_With_Gatekeeper
Voting_Without_Gatekeeper
Stacking_LR_Meta_Passthrough_False
Stacking_RF_Meta_Passthrough_True
Top4_FeatureSet_EqualWeight
MultiView_EqualWeight
MultiView_LinearMeta
```

Bu adımda amaç, tek bir RF gatekeeper'ın üstüne ensemble, stacking ve multiview yaklaşımlarının gerçek bir performans artışı sağlayıp sağlamadığını kontrol etmektir.

Karşılaştırma mantığı:

```text
gatekeeper RF sonucu
vs
en iyi advanced ensemble sonucu
```

Sonuçlar iki pipeline için ayrı ayrı raporlanır.

Çıktı klasörü:

```text
molfest_outputs/08_advanced_ensembles/
```

---

### 09_collect_best_candidates.ipynb

Bu notebook final toplama ve raporlama adımıdır.

Model eğitmez. Önceki adımların CSV çıktılarını okur.

Okuma sırası:

```text
önce local molfest_outputs/
eğer localde yoksa GitHub raw molfest_outputs/
```

Bu nedenle GitHub'a `molfest_outputs/` klasörü yüklendikten sonra Colab'da doğrudan çalışabilir.

Bu notebook şunları yapar:

```text
03-08 arasındaki sonuçları toplar
02_RF_Morgan_Baseline satırını başlangıç baseline olarak ekler
bütün adayları ROC-AUC değerine göre sıralar
her adım için en iyi adayı seçer
pipeline progression grafiği çizer
final seçilen adayları raporlar
```

Progression başlangıcı:

```text
02_rf_morgan_baseline = RandomForest + Morgan fingerprint + no sampling
```

Progression figürü ROC-AUC eksenini 0.70'ten başlatır. Böylece küçük performans farkları daha görünür hale gelir.

Çıktı klasörü:

```text
molfest_outputs/09_collect_best_candidates/
```

---

## 5. Temel metrikler

Confusion matrix bileşenleri:

```text
TP = True Positive
TN = True Negative
FP = False Positive
FN = False Negative
```

Metrikler:

```text
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
Precision = TP / (TP + FP)
F1 = 2 × Precision × Recall / (Precision + Recall)
Balanced Accuracy = (Recall + Specificity) / 2
```

ROC-AUC, modelin sınıfları ayırma gücünü özetler.

AP, pozitif sınıfı yakalama kalitesini precision-recall mantığıyla özetler.

Bu pipeline'da ana karar metriği çoğunlukla:

```text
ROC-AUC
```

Ek kontrol metrikleri:

```text
AP
F1
Balanced Accuracy
Recall
Specificity
Precision
```

---

## 6. Önerilen anlatım sırası

Toplam 5 derslik akış için önerilen dağılım:

```text
Ders 1:
01_generate_feature_store.ipynb
02_rf_morgan_baseline.ipynb

Ders 2:
03_rf_feature_ablation.ipynb
04_feature_selection.ipynb

Ders 3:
05_train_12_models_to_search_a_model.ipynb
06_resampling_search.ipynb

Ders 4:
07_random_search_tuning_test_performance_candidates_fixed.ipynb
08_advanced_architectures.ipynb

Ders 5:
09_collect_best_candidates.ipynb
genel değerlendirme ve pipeline yorumu
```

---

## 7. Colab kullanımı

Notebooku Colab'da açtıktan sonra:

```text
Runtime → Run all
```

veya Türkçe arayüzde:

```text
Çalışma zamanı → Tümünü çalıştır
```

Feature üretimi uzun sürebileceği için `01_generate_feature_store.ipynb` genellikle bir kere çalıştırılır. Sonraki notebooklar hazır feature CSV dosyalarını ve `molfest_outputs/` içindeki sonuç dosyalarını kullanır.

---

## 8. Local çalışma

Localde çalıştırmak için örnek:

```bash
cd /path/to/MOL_FEST_2026
python3 01_generate_feature_store.py
python3 02_rf_morgan_baseline.py
python3 03_rf_feature_ablation.py
python3 04_feature_selection.py
python3 05_train_12_models_to_search_a_model.py
python3 06_resampling_search.py
python3 07_random_search_tuning_test_performance_candidates_fixed.py
python3 08_advanced_architectures.py
python3 09_collect_best_candidates.py
```

Tüm sonuçlar `molfest_outputs/` altında birikir.

---

## 9. GitHub'a yüklenecek temel çıktılar

GitHub'a en az şu klasörler yüklenmelidir:

```text
molfest_outputs/01_feature_store/
molfest_outputs/03_rf_feature_ablation/
molfest_outputs/04_feature_selection/
molfest_outputs/05_train_12_models/
molfest_outputs/06_resampling/
molfest_outputs/07_random_search_tuning_fast/
molfest_outputs/08_advanced_ensembles/
molfest_outputs/09_collect_best_candidates/
```

Böylece Colab notebookları local çıktı klasörü olmasa bile GitHub raw linklerinden sonuçları okuyabilir.

---

## 10. Kısa pipeline özeti

```text
01: Ham veri → temiz veri → Morgan/MACCS/Avalon/RDKit feature store
02: RF + Morgan baseline
03: RF ile feature set ablation
04: Feature selection denemeleri
05: 12 model karşılaştırması
06: RF ile over/under sampling
07: RF random search tuning ve test-set candidate tablosu
08: Voting / stacking / multiview advanced ensemble yapıları
09: Tüm adayları topla, ROC-AUC sıralı raporla, final progression çiz
```
