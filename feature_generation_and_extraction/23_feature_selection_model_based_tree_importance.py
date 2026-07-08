#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-based feature selection with RandomForest and ExtraTrees importances."""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from common_feature_selection_utils import load_matrix, save_selection, k_list, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; POSITIVE_THRESHOLD=None
K_VALUES=[50,100,150,200]; OUTDIR='feature_outputs/23_model_based_tree_importance'
RANDOM_STATE=42; N_ESTIMATORS=500; N_JOBS=-1

def main():
    outdir=ensure_dir(OUTDIR); mdir=ensure_dir(outdir/'saved_selector_models')
    df,X,y,cols,target,smiles,path,dropped=load_matrix(INPUT_FILE,CSV_SEPARATOR,TARGET_COLUMN,SMILES_COLUMN,POSITIVE_THRESHOLD)
    models={
        'rf_importance':RandomForestClassifier(n_estimators=N_ESTIMATORS,max_features='sqrt',class_weight='balanced_subsample',random_state=RANDOM_STATE,n_jobs=N_JOBS),
        'extratrees_importance':ExtraTreesClassifier(n_estimators=N_ESTIMATORS,max_features='sqrt',class_weight='balanced',random_state=RANDOM_STATE,n_jobs=N_JOBS)}
    for name,model in models.items():
        print('\nTraining',name); model.fit(X.values,y); joblib.dump({'model':model,'feature_cols':cols},mdir/f'{name}.joblib')
        ranking=pd.DataFrame({'Feature':cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False).reset_index(drop=True); ranking['Rank']=np.arange(1,len(ranking)+1)
        ks=k_list(len(cols),K_VALUES); selected={k:ranking['Feature'].head(k).tolist() for k in ks}
        save_selection(outdir,name,df,X,y,target,smiles,ranking,selected); print(ranking.head(max(ks)).to_string(index=False))
    print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
