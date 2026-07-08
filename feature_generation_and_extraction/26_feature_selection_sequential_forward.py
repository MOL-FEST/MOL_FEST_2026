#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential Forward Selection with Logistic Regression. Slow; uses ANOVA prefilter."""
import numpy as np, pandas as pd, joblib
from sklearn.feature_selection import f_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common_feature_selection_utils import load_matrix, save_selection, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
PREFILTER_TOP_N=120; MAX_SELECTED_FEATURES=50; CV=3; SCORING='roc_auc'; N_JOBS=-1; RANDOM_STATE=42; OUTDIR='feature_outputs/26_sequential_forward'

def main():
    outdir=ensure_dir(OUTDIR); df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    F,_=f_classif(X.values,y); F=np.nan_to_num(F,nan=0.0,posinf=0.0,neginf=0.0); order=np.argsort(F)[::-1][:min(PREFILTER_TOP_N,X.shape[1])]
    pre=[cols[i] for i in order]; Xpre=X[pre].values; nsel=min(MAX_SELECTED_FEATURES,len(pre))
    est=Pipeline([('scaler',StandardScaler()),('logreg',LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=5000,random_state=RANDOM_STATE))])
    sfs=SequentialFeatureSelector(est,n_features_to_select=nsel,direction='forward',scoring=SCORING,cv=CV,n_jobs=N_JOBS)
    print(f'Running SFS on {len(pre)} features; selecting {nsel}.'); sfs.fit(Xpre,y)
    sel=[f for f,keep in zip(pre,sfs.get_support()) if keep]; ranking=pd.DataFrame({'Feature':sel+[f for f in pre if f not in set(sel)]})
    ranking['sfs_selected']=ranking['Feature'].isin(sel); ranking['Rank']=np.arange(1,len(ranking)+1)
    joblib.dump({'sfs':sfs,'prefilter_features':pre,'selected_features':sel},outdir/'sequential_forward_selector.joblib')
    save_selection(outdir,'sequential_forward',df,X,y,target,smiles,ranking,{nsel:sel})
    print(ranking.head(nsel).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
