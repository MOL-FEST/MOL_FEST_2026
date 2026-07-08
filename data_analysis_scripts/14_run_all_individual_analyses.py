#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_run_all_individual_analyses.py
=================================

Runs every visual-analysis script one by one.

Before running:
- Put this file, common_data_features.py, and all analysis scripts in the same folder.
- Put the input CSV in the same folder or edit INPUT_FILE inside each script.

Usage:
  python 14_run_all_individual_analyses.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "04_anova_fscore_top30.py",
    "05_correlation_heatmap_top30.py",
    "06_feature_dendrogram_top30.py",
    "07_pca_projection.py",
    "08_tsne_projection.py",
    "09_umap_projection.py",
    "10_radviz_top30.py",
    "11_shap_beeswarm_bar_dependence.py",
    "12_shap_waterfall_cases.py",
    "13_lime_local_explanations.py",
]


def main():
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print("\n" + "=" * 90)
        print(f"RUNNING: {script}")
        print("=" * 90)
        subprocess.run([sys.executable, str(here / script)], check=True)

    print("\nAll visual analyses completed.")


if __name__ == "__main__":
    main()
