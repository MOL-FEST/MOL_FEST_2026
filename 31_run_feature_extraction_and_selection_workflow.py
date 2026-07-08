#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run recommended feature extraction and selection workflow."""
import subprocess, sys
from pathlib import Path
FAST=['18_feature_extraction_fingerprints.py','29_data_quality_leakage_check.py','19_feature_selection_variance_threshold.py','20_feature_selection_anova_f_classif.py','21_feature_selection_chi2.py','22_feature_selection_mutual_info.py','23_feature_selection_model_based_tree_importance.py','24_feature_selection_l1_logistic.py','30_compare_selected_feature_sets.py']
SLOW=['25_feature_selection_rfe_logistic.py','26_feature_selection_sequential_forward.py','27_feature_selection_permutation_importance.py','28_dimensionality_reduction_pca.py']
RUN_SLOW_OPTIONAL=False

def run(p): print('\n'+'='*90+'\nRUNNING: '+p.name+'\n'+'='*90); subprocess.run([sys.executable,str(p)],check=True)
def main():
    here=Path(__file__).resolve().parent
    for s in FAST: run(here/s)
    if RUN_SLOW_OPTIONAL:
        for s in SLOW: run(here/s)
    else: print('\nSlow optional scripts skipped. Set RUN_SLOW_OPTIONAL=True to run RFE/SFS/permutation/PCA.')
    print('\n[DONE] Workflow completed.')
if __name__=='__main__': main()
