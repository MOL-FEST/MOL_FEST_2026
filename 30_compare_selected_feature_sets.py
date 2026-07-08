#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare selected feature sets across feature-selection methods."""
from pathlib import Path
import numpy as np, pandas as pd
from common_feature_selection_utils import ensure_dir
SELECTOR_OUTPUT_DIRS=['feature_outputs/20_anova_f_classif','feature_outputs/21_chi2','feature_outputs/22_mutual_info','feature_outputs/23_model_based_tree_importance','feature_outputs/24_l1_logistic']
K=100; OUTDIR='feature_outputs/30_compare_feature_sets'

def load_sets():
    sets={}
    for folder in SELECTOR_OUTPUT_DIRS:
        for f in Path(folder).glob('*_selection_summary.csv'):
            s=pd.read_csv(f); s=s[s['k']==K]
            for _,r in s.iterrows():
                sets[str(r['method'])]=set(str(r['selected_features']).split(';')) if pd.notna(r['selected_features']) else set()
    return sets

def main():
    outdir=ensure_dir(OUTDIR); sets=load_sets()
    if not sets: raise ValueError('No selection summaries found. Run selection scripts first.')
    methods=sorted(sets); overlap=pd.DataFrame(index=methods,columns=methods,dtype=int); jac=pd.DataFrame(index=methods,columns=methods,dtype=float)
    for a in methods:
        for b in methods:
            inter=len(sets[a]&sets[b]); union=len(sets[a]|sets[b]); overlap.loc[a,b]=inter; jac.loc[a,b]=inter/union if union else 0
    overlap.to_csv(outdir/'selected_feature_overlap_matrix.csv'); jac.to_csv(outdir/'selected_feature_jaccard_matrix.csv')
    allf=sorted(set().union(*sets.values())); mem=pd.DataFrame({'Feature':allf})
    for m in methods: mem[m]=mem['Feature'].isin(sets[m]).astype(int)
    mem['n_methods_selected']=mem[methods].sum(axis=1); mem.sort_values(['n_methods_selected','Feature'],ascending=[False,True]).to_csv(outdir/'selected_feature_membership_table.csv',index=False)
    print('Overlap matrix:'); print(overlap.to_string()); print('\nJaccard matrix:'); print(jac.round(3).to_string()); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
