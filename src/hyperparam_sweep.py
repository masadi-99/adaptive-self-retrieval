"""Hyperparameter sensitivity: sweep alpha, beta, k on 3 datasets at H=96."""
import os, sys, numpy as np, pandas as pd, faiss, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor
from fusion import RCAF

extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')

def build_kb(data, ws, pl, stride=64):
    w, f, e = [], [], []
    for s in range(0, len(data)-ws-pl, stride):
        ww, ff = data[s:s+ws], data[s+ws:s+ws+pl]
        if len(ff)<pl: continue
        w.append(ww); f.append(ff); e.append(extractor.extract_embedding(ww))
    idx = faiss.IndexFlatL2(extractor.embedding_dim)
    if e: idx.add(np.stack(e).astype(np.float32))
    return w, f, idx

def eval_one(test, ws, pl, kb_w, kb_f, idx, k, alpha, beta):
    positions = list(range(0, len(test)-ws-pl+1, pl))
    uncs = [extractor.predict_with_uncertainty(test[s:s+ws], pl)[2] for s in positions]
    um, us = np.mean(uncs), np.std(uncs)+1e-8
    rcaf = RCAF(extractor, k=k, alpha=alpha, beta=beta)
    bm, rm = [], []
    for s in positions:
        ctx, gt = test[s:s+ws], test[s+ws:s+ws+pl]
        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        bm.append(float(np.mean((med-gt)**2)))
        fc = rcaf.fuse(med, unc, None, pl, current_context=ctx,
                       kb_windows=kb_w, kb_futures=kb_f, faiss_index=idx,
                       unc_mean=um, unc_std=us)
        rm.append(float(np.mean((fc-gt)**2)))
    return np.mean(bm), np.mean(rm)

rows = []
for ds in ['ETTm2', 'Weather', 'ECL']:
    print(f"\n{'='*40}\n{ds}")
    train, val, test, config = load_dataset(ds, '../data')
    tv = np.concatenate([train, val])
    kb_w, kb_f, idx = build_kb(tv, 512, 96)
    bl, _ = eval_one(test, 512, 96, kb_w, kb_f, idx, 7, 0, 0)
    for sweep, vals, fa, fb, fk in [
        ('alpha', [0.1,0.2,0.35,0.5,0.7], None, 0.25, 7),
        ('beta',  [0.1,0.2,0.25,0.35,0.5], 0.35, None, 7),
        ('k',     [3,5,7,10],               0.35, 0.25, None),
    ]:
        print(f"  {sweep} sweep:")
        for v in vals:
            a = v if sweep=='alpha' else fa
            b = v if sweep=='beta' else fb
            k = v if sweep=='k' else fk
            _, rc = eval_one(test, 512, 96, kb_w, kb_f, idx, k, a, b)
            imp = (bl-rc)/bl*100
            print(f"    {sweep}={v}: {imp:+.2f}%")
            rows.append({'dataset':ds, 'sweep':sweep, 'param':v, 'improvement':imp})

pd.DataFrame(rows).to_csv('../results/hyperparam_sweep.csv', index=False)
print("\nSaved results/hyperparam_sweep.csv")
