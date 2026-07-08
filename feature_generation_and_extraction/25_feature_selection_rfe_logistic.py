#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recursive Feature Elimination with Logistic Regression. Slow; uses ANOVA prefilter."""
import numpy as np, pandas as pd, joblib
from sklearn.feature_selection import f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; PREFILTER_TOP_N=500; OUTDIR='feature_outputs/25_rfe_logistic'; RANDOM_STATE=42

def main():
    outdir=ensure_dir(OUTDIR); df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    F,_=f_classif(X.values,y); F=np.nan_to_num(F,nan=0.0,posinf=0.0,neginf=0.0); order=np.argsort(F)[::-1][:min(PREFILTER_TOP_N,X.shape[1])]
    pre=[cols[i] for i in order]; Xpre=X[pre].values; maxk=min(max(K_VALUES),len(pre))
    scaler=StandardScaler(); Xs=scaler.fit_transform(Xpre); est=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=5000,random_state=RANDOM_STATE)
    rfe=RFE(estimator=est,n_features_to_select=maxk,step=0.1); rfe.fit(Xs,y)
    sel=[f for f,keep in zip(pre,rfe.support_) if keep]; final=Pipeline([('scaler',StandardScaler()),('logreg',est)]); final.fit(X[sel].values,y)
    coef=np.abs(final.named_steps['logreg'].coef_.ravel()); ranking=pd.DataFrame({'Feature':sel,'abs_final_coefficient':coef,'rfe_selected':True}).sort_values('abs_final_coefficient',ascending=False)
    rest=pd.DataFrame({'Feature':[f for f in pre if f not in set(sel)],'abs_final_coefficient':0.0,'rfe_selected':False})
    ranking=pd.concat([ranking,rest]).reset_index(drop=True); ranking['Rank']=np.arange(1,len(ranking)+1)
    joblib.dump({'scaler':scaler,'rfe':rfe,'final_pipe':final,'prefilter_features':pre},outdir/'rfe_logistic_selector.joblib')
    ks=k_list(len(ranking),K_VALUES); selected={k:ranking['Feature'].head(k).tolist() for k in ks}; save_selection(outdir,'rfe_logistic',df,X,y,target,smiles,ranking,selected)
    print(ranking.head(max(ks)).to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
