#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutation importance feature selection using held-out ROC-AUC drop."""
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; RANDOM_STATE=42; TEST_SIZE=0.2; N_ESTIMATORS=300; N_REPEATS=10; SCORING='roc_auc'; PREFILTER_TOP_N=300; N_JOBS=-1; OUTDIR='feature_outputs/27_permutation_importance'

def main():
    outdir=ensure_dir(OUTDIR); df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    Xtr,Xte,ytr,yte=train_test_split(X.values,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    rf=RandomForestClassifier(n_estimators=N_ESTIMATORS,max_features='sqrt',class_weight='balanced_subsample',random_state=RANDOM_STATE,n_jobs=N_JOBS).fit(Xtr,ytr)
    order=np.argsort(rf.feature_importances_)[::-1][:min(PREFILTER_TOP_N,X.shape[1])]; pre=[cols[i] for i in order]
    rf2=RandomForestClassifier(n_estimators=N_ESTIMATORS,max_features='sqrt',class_weight='balanced_subsample',random_state=RANDOM_STATE,n_jobs=N_JOBS).fit(Xtr[:,order],ytr)
    res=permutation_importance(rf2,Xte[:,order],yte,scoring=SCORING,n_repeats=N_REPEATS,random_state=RANDOM_STATE,n_jobs=N_JOBS)
    ranking=pd.DataFrame({'Feature':pre,'permutation_importance_mean':res.importances_mean,'permutation_importance_sd':res.importances_std}).sort_values('permutation_importance_mean',ascending=False).reset_index(drop=True); ranking['Rank']=np.arange(1,len(ranking)+1)
    joblib.dump({'rf_full_prefilter':rf,'rf_prefiltered':rf2,'prefilter_features':pre},outdir/'permutation_importance_selector_models.joblib')
    ks=k_list(len(ranking),K_VALUES); selected={k:ranking['Feature'].head(k).tolist() for k in ks}; save_selection(outdir,'permutation_importance',df,X,y,target,smiles,ranking,selected)
    print(ranking.head(max(ks)).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
