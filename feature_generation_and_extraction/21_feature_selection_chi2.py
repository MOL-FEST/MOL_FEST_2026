#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised univariate selection with chi-square test."""
import numpy as np, pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; OUTDIR='feature_outputs/21_chi2'


def main():
    outdir=ensure_dir(OUTDIR)
    df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)

    Xv=X.values.astype(float); Xc=MinMaxScaler().fit_transform(Xv) if np.nanmin(Xv)<0 else Xv
    s,p=chi2(Xc,y); s=np.nan_to_num(s,nan=0.0,posinf=0.0,neginf=0.0); p=np.nan_to_num(p,nan=1.0,posinf=1.0,neginf=1.0)
    ranking=pd.DataFrame({'Feature':cols,'chi2_score':s,'p_value':p}).sort_values('chi2_score',ascending=False).reset_index(drop=True)
    ranking['Rank']=np.arange(1,len(ranking)+1)
    kvalues=k_list(len(cols),K_VALUES); selected={k: ranking['Feature'].head(k).tolist() for k in kvalues}
    save_selection(outdir,'chi2',df,X,y,target,smiles,ranking,selected)
    print(ranking.head(max(kvalues)).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
