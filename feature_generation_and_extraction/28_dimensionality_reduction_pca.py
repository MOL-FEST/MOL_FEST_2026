#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCA dimensionality reduction; contrast with original-feature selection."""
import numpy as np, pandas as pd, joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common_feature_selection_utils import load_matrix, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
N_COMPONENTS_LIST=[50,100,150,200]; RANDOM_STATE=42; OUTDIR='feature_outputs/28_pca_reduction'

def main():
    outdir=ensure_dir(OUTDIR); df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    ks=k_list(min(X.shape[0]-1,X.shape[1]),N_COMPONENTS_LIST); keep=[]
    if smiles and smiles in df.columns: keep.append(smiles)
    if target in df.columns: keep.append(target)
    rows=[]
    for k in ks:
        pipe=Pipeline([('scaler',StandardScaler()),('pca',PCA(n_components=k,random_state=RANDOM_STATE))]); Xp=pipe.fit_transform(X.values)
        out=pd.concat([df[keep].reset_index(drop=True),pd.DataFrame(Xp,columns=[f'PC_{i+1}' for i in range(k)])],axis=1)
        out.to_csv(outdir/f'pca_{k}_components.csv',index=False); joblib.dump({'pipeline':pipe,'input_feature_cols':cols},outdir/f'pca_model_{k}.joblib')
        evr=pipe.named_steps['pca'].explained_variance_ratio_; rows.append({'n_components':k,'cumulative_explained_variance':float(evr.sum()),'first_component_explained_variance':float(evr[0])})
    pd.DataFrame(rows).to_csv(outdir/'pca_explained_variance_summary.csv',index=False); print(pd.DataFrame(rows).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
