# Individual visual-analysis scripts for classification workshop

Default input:

```python
INPUT_FILE = "A15_A18_ERa_LUC_VM7_agonist_antagonist.csv"
CSV_SEPARATOR = ";"
TARGET_COLUMN = "binary_label_agonist1_antagonist0"
SMILES_COLUMN = "QSAR-Ready SMILES"
FEATURE_MODE = "smiles"
FINGERPRINT_TYPE = "maccs"
```

For the ChEMBL-prepared dataset from `03_fetch_prepare_chembl_target_dataset.py`, set:

```python
INPUT_FILE = "CHEMBL206_prepared_molecule_classification.csv"
CSV_SEPARATOR = ","
TARGET_COLUMN = "binary_label_active1_inactive0"
SMILES_COLUMN = "QSAR-Ready SMILES"
```

Scripts:

- `04_anova_fscore_top30.py`
- `05_correlation_heatmap_top30.py`
- `06_feature_dendrogram_top30.py`
- `07_pca_projection.py`
- `08_tsne_projection.py`
- `09_umap_projection.py`
- `10_radviz_top30.py`
- `11_shap_beeswarm_bar_dependence.py`
- `12_shap_waterfall_cases.py`
- `13_lime_local_explanations.py`
- `14_run_all_individual_analyses.py`

Install:

```bash
pip install -r requirements_visual_analyses.txt
```

Run individually:

```bash
python 04_anova_fscore_top30.py
python 11_shap_beeswarm_bar_dependence.py
python 12_shap_waterfall_cases.py
```

Run all:

```bash
python 14_run_all_individual_analyses.py
```
