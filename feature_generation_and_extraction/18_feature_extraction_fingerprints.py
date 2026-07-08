#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate MACCS, Morgan and Avalon fingerprints from a SMILES CSV."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from common_feature_selection_utils import resolve, read_table, detect_smiles, detect_target, ensure_dir
INPUT_FILE='A15_A18_ERa_LUC_VM7_agonist_antagonist.csv'; CSV_SEPARATOR=';'
SMILES_COLUMN='QSAR-Ready SMILES'; TARGET_COLUMN='binary_label_agonist1_antagonist0'
OUTDIR='feature_outputs/18_fingerprints'
MAKE_MACCS=True; MAKE_MORGAN=True; MAKE_AVALON=True
MORGAN_BITS=1024; MORGAN_RADIUS=2; AVALON_BITS=1024
KEEP_ORIGINAL_COLUMNS=True; DROP_INVALID_SMILES=True

def mol_from_smiles(s):
    from rdkit import Chem
    m=Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    return (Chem.MolToSmiles(m,canonical=True,isomericSmiles=True),m) if m is not None else (None,None)
def maccs(m):
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys
    fp=MACCSkeys.GenMACCSKeys(m); arr=np.zeros((167,),dtype=np.int8); DataStructs.ConvertToNumpyArray(fp,arr)
    return arr[1:],[f'MACCS_{i}' for i in range(1,167)]
def morgan(m):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS,fpSize=MORGAN_BITS)
    fp=gen.GetFingerprint(m); arr=np.zeros((MORGAN_BITS,),dtype=np.int8); DataStructs.ConvertToNumpyArray(fp,arr)
    return arr,[f'Morgan_r{MORGAN_RADIUS}_{i}' for i in range(MORGAN_BITS)]
def avalon(m):
    from rdkit import DataStructs
    from rdkit.Avalon import pyAvalonTools
    fp=pyAvalonTools.GetAvalonFP(m,nBits=AVALON_BITS); arr=np.zeros((AVALON_BITS,),dtype=np.int8); DataStructs.ConvertToNumpyArray(fp,arr)
    return arr,[f'Avalon_{i}' for i in range(AVALON_BITS)]
def block(mols,fn):
    rows=[]; names=None
    for m in mols:
        a,names=fn(m); rows.append(a)
    return pd.DataFrame(np.vstack(rows),columns=names),names

def main():
    outdir=ensure_dir(OUTDIR); path=resolve(INPUT_FILE); df=read_table(path,CSV_SEPARATOR)
    smi_col=detect_smiles(df) if SMILES_COLUMN in [None,'AUTO'] else SMILES_COLUMN
    tgt_col=detect_target(df) if TARGET_COLUMN in [None,'AUTO'] else TARGET_COLUMN
    can=[]; mols=[]; valid=[]
    for s in df[smi_col]:
        c,m=mol_from_smiles(s); can.append(c); mols.append(m); valid.append(m is not None)
    valid=np.array(valid,dtype=bool); invalid=int((~valid).sum())
    if DROP_INVALID_SMILES:
        df2=df.loc[valid].copy().reset_index(drop=True); mols=[m for m,v in zip(mols,valid) if v]; can=[c for c,v in zip(can,valid) if v]
    else: df2=df.copy().reset_index(drop=True)
    df2['canonical_smiles_rdkit']=can
    blocks=[]; manifest=[]; report=[]
    if MAKE_MACCS:
        b,n=block(mols,maccs); blocks.append(b); manifest += [{'feature':x,'family':'MACCS'} for x in n]; report.append({'family':'MACCS','n_features':len(n),'status':'created'})
    if MAKE_MORGAN:
        b,n=block(mols,morgan); blocks.append(b); manifest += [{'feature':x,'family':'Morgan'} for x in n]; report.append({'family':'Morgan','n_features':len(n),'status':'created'})
    if MAKE_AVALON:
        try:
            b,n=block(mols,avalon); blocks.append(b); manifest += [{'feature':x,'family':'Avalon'} for x in n]; report.append({'family':'Avalon','n_features':len(n),'status':'created'})
        except Exception as e:
            print('[WARNING] Avalon skipped:',e); report.append({'family':'Avalon','n_features':0,'status':f'skipped: {e}'})
    feats=pd.concat(blocks,axis=1)
    out=pd.concat([df2.reset_index(drop=True),feats],axis=1) if KEEP_ORIGINAL_COLUMNS else pd.concat([df2[[c for c in [smi_col,'canonical_smiles_rdkit',tgt_col] if c in df2]].reset_index(drop=True),feats],axis=1)
    out.to_csv(outdir/'molecules_with_fingerprints.csv',index=False); df2.to_csv(outdir/'cleaned_molecules.csv',index=False)
    pd.DataFrame(manifest).to_csv(outdir/'feature_manifest.csv',index=False)
    rep=pd.DataFrame(report); rep.loc[len(rep)]={'family':'TOTAL','n_features':feats.shape[1],'status':'created'}; rep.loc[len(rep)]={'family':'INVALID_SMILES_REMOVED','n_features':invalid,'status':'count'}
    rep.to_csv(outdir/'feature_extraction_report.csv',index=False)
    print(rep.to_string(index=False)); print('[DONE]',(outdir/'molecules_with_fingerprints.csv').resolve())
if __name__=='__main__': main()
