#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse model-based feature selection using L1 Logistic Regression."""
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; OUTDIR='feature_outputs/24_l1_logistic'; RANDOM_STATE=42; C_VALUE=1.0

def main():
    outdir=ensure_dir(OUTDIR); df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    model=Pipeline([('scaler',StandardScaler()),('logreg',LogisticRegression(penalty='l1',solver='liblinear',C=C_VALUE,class_weight='balanced',max_iter=5000,random_state=RANDOM_STATE))])
    model.fit(X.values,y); joblib.dump({'model':model,'feature_cols':cols},outdir/'l1_logistic_selector.joblib')
    coef=model.named_steps['logreg'].coef_.ravel(); ranking=pd.DataFrame({'Feature':cols,'coefficient':coef,'abs_coefficient':np.abs(coef),'selected_nonzero':np.abs(coef)>0}).sort_values('abs_coefficient',ascending=False).reset_index(drop=True); ranking['Rank']=np.arange(1,len(ranking)+1)
    ks=k_list(len(cols),K_VALUES); selected={k:ranking['Feature'].head(k).tolist() for k in ks}; save_selection(outdir,'l1_logistic',df,X,y,target,smiles,ranking,selected)
    print(ranking.head(max(ks)).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
