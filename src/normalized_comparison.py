"""S3: RCAF on normalized data for direct TS-RAG comparison."""
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

tsrag = {'ETTh1':{'base':0.3920,'tsrag':0.3557},'ETTh2':{'base':0.2982,'tsrag':0.2451},
         'ETTm1':{'base':0.3506,'tsrag':0.2906},'ETTm2':{'base':0.1964,'tsrag':0.1466},
         'Weather':{'base':0.1801,'tsrag':0.1454},'ECL':{'base':0.1967,'tsrag':0.1120}}

rows = []
for ds in ['ETTh1','ETTm1','ETTm2','Weather','ECL']:
    train_raw, val_raw, test_raw, config = load_dataset(ds, '../data')
    tm, ts = train_raw.mean(), train_raw.std()
    train, val, test = (train_raw-tm)/ts, (val_raw-tm)/ts, (test_raw-tm)/ts
    tv = np.concatenate([train, val])
    ds_bl, ds_rc = [], []
    for pl in config['prediction_lengths']:
        print(f"  Norm: {ds} H={pl}")
        kb_w, kb_f, idx = build_kb(tv, 512, pl)
        positions = list(range(0, len(test)-512-pl+1, pl))
        uncs = [extractor.predict_with_uncertainty(test[s:s+512], pl)[2] for s in positions]
        um, us = np.mean(uncs), np.std(uncs)+1e-8
        rcaf_m = RCAF(extractor, k=7, alpha=0.35, beta=0.25)
        bm, rm = [], []
        for s in positions:
            ctx, gt = test[s:s+512], test[s+512:s+512+pl]
            med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
            bm.append(float(np.mean((med-gt)**2)))
            fc = rcaf_m.fuse(med, unc, None, pl, current_context=ctx,
                            kb_windows=kb_w, kb_futures=kb_f, faiss_index=idx,
                            unc_mean=um, unc_std=us)
            rm.append(float(np.mean((fc-gt)**2)))
        bl, rc = np.mean(bm), np.mean(rm)
        imp = (bl-rc)/bl*100
        print(f"    base={bl:.4f} RCAF={rc:.4f} ({imp:+.2f}%)")
        rows.append({'dataset':ds,'pred_len':pl,'norm_base':bl,'norm_rcaf':rc,'improvement':imp})
        ds_bl.append(bl); ds_rc.append(rc)
    avg_imp = (np.mean(ds_bl)-np.mean(ds_rc))/np.mean(ds_bl)*100
    print(f"  {ds} avg: {avg_imp:+.2f}%")

pd.DataFrame(rows).to_csv('../results/normalized_comparison.csv', index=False)
df = pd.DataFrame(rows)
print(f"\n{'='*75}")
print("TS-RAG vs RCAF (normalized MSE)")
print(f"{'='*75}")
print(f"{'DS':10s} {'TS-RAG Δ':>10s} {'RCAF Δ':>10s}")
for ds in ['ETTh1','ETTm1','ETTm2','Weather','ECL']:
    sub = df[df.dataset==ds]
    ri = (sub.norm_base.mean()-sub.norm_rcaf.mean())/sub.norm_base.mean()*100
    ti = tsrag.get(ds,{})
    ti_imp = (ti.get('base',0)-ti.get('tsrag',0))/ti.get('base',1)*100 if ti else 0
    print(f"{ds:10s} {ti_imp:>+9.1f}% {ri:>+9.1f}%")
