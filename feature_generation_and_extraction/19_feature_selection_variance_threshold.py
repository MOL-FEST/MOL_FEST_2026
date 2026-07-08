#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsupervised selection: rank features by variance."""
import numpy as np, pandas as pd

from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; OUTDIR='feature_outputs/19_variance_threshold'


def main():
    outdir=ensure_dir(OUTDIR)
    df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)

    ranking=pd.DataFrame({'Feature':cols,'Variance':X.var(axis=0).values}).sort_values('Variance',ascending=False).reset_index(drop=True)
    ranking['Rank']=np.arange(1,len(ranking)+1)
    pd.DataFrame({'DroppedConstantFeature':dropped}).to_csv(outdir/'dropped_constant_features.csv',index=False)
    kvalues=k_list(len(cols),K_VALUES); selected={k: ranking['Feature'].head(k).tolist() for k in kvalues}
    save_selection(outdir,'variance_threshold',df,X,y,target,smiles,ranking,selected)
    print(ranking.head(max(kvalues)).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
