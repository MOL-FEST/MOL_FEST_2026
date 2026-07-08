#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic data-quality and leakage checks for molecular classification."""
import numpy as np, pandas as pd
from common_feature_selection_utils import resolve, read_table, detect_target, detect_smiles, feature_cols, ensure_dir
INPUT_FILE='feature_outputs/18_fingerprints/molecules_with_fingerprints.csv'; CSV_SEPARATOR=','
TARGET_COLUMN='binary_label_agonist1_antagonist0'; SMILES_COLUMN='QSAR-Ready SMILES'; OUTDIR='feature_outputs/29_data_quality_leakage_check'
CORR_PREFILTER_TOP_N=300; HIGH_CORR_THRESHOLD=0.95

def valid_smiles(s):
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(str(s)) is not None
    except Exception: return None

def main():
    outdir=ensure_dir(OUTDIR); path=resolve(INPUT_FILE); df=read_table(path,CSV_SEPARATOR)
    target=detect_target(df) if TARGET_COLUMN in [None,'AUTO'] else TARGET_COLUMN
    try: smiles=detect_smiles(df) if SMILES_COLUMN in [None,'AUTO'] else SMILES_COLUMN
    except Exception: smiles=None
    rep=[{'check':'input_file','value':str(path)},{'check':'n_rows','value':len(df)},{'check':'n_columns','value':df.shape[1]},{'check':'target_column','value':target},{'check':'smiles_column','value':smiles}]
    y=pd.to_numeric(df[target],errors='coerce'); rep.append({'check':'missing_target_rows','value':int(y.isna().sum())})
    for cls,cnt in y.dropna().value_counts().sort_index().items(): rep.append({'check':f'class_count_{cls}','value':int(cnt)})
    if smiles and smiles in df.columns:
        dup=df[df[smiles].duplicated(keep=False)].copy(); dup.to_csv(outdir/'duplicate_smiles.csv',index=False); rep.append({'check':'duplicate_smiles_rows','value':len(dup)})
        val=df[smiles].map(valid_smiles); rep.append({'check':'invalid_smiles_rows','value':int((val==False).sum())})
    keys=['label','class','target','activity','standard_value','pchembl','relation','comment','agonist','antagonist','active','inactive']
    susp=[{'column':c,'reason':'target-like/leakage-like name'} for c in df.columns if c!=target and any(k in c.lower() for k in keys)]
    pd.DataFrame(susp).to_csv(outdir/'suspicious_columns.csv',index=False); rep.append({'check':'suspicious_column_count','value':len(susp)})
    cols=feature_cols(df,target,smiles); rep.append({'check':'detected_feature_count','value':len(cols)})
    if cols:
        X=df[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0)
        std=X.std(axis=0); const=list(std[std==0].index); pd.DataFrame({'constant_feature':const}).to_csv(outdir/'constant_features.csv',index=False); rep.append({'check':'constant_feature_count','value':len(const)})
        top=std.sort_values(ascending=False).head(min(CORR_PREFILTER_TOP_N,len(std))).index.tolist(); corr=X[top].corr().fillna(0.0).abs(); pairs=[]
        for i,a in enumerate(top):
            for b in top[i+1:]:
                v=float(corr.loc[a,b])
                if v>=HIGH_CORR_THRESHOLD: pairs.append({'feature_1':a,'feature_2':b,'abs_corr':v})
        pd.DataFrame(pairs).sort_values('abs_corr',ascending=False if pairs else True).to_csv(outdir/'high_correlation_pairs.csv',index=False); rep.append({'check':'high_correlation_pair_count','value':len(pairs)})
    report=pd.DataFrame(rep); report.to_csv(outdir/'data_quality_report.csv',index=False); print(report.to_string(index=False)); print('[DONE]',outdir.resolve())
if __name__=='__main__': main()
