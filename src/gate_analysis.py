"""S4: Does consensus agreement predict RCAF improvement?"""
import os, sys, numpy as np, pandas as pd, faiss, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor

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

all_rows = []
for ds in ['ETTm2','Weather','ETTh1','ECL']:
    train, val, test, _ = load_dataset(ds, '../data')
    tv = np.concatenate([train, val]); pl = 96
    print(f"\n  Gate: {ds} H={pl}")
    kb_w, kb_f, idx = build_kb(tv, 512, pl)
    positions = list(range(0, len(test)-512-pl+1, pl))
    uncs = [extractor.predict_with_uncertainty(test[s:s+512], pl)[2] for s in positions]
    um, us = np.mean(uncs), np.std(uncs)+1e-8
    for start in positions:
        ctx, gt = test[start:start+512], test[start+512:start+512+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        base_mse = float(np.mean((med-gt)**2))
        if idx.ntotal < 5: continue
        emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
        D, I = idx.search(emb, 7)
        nn_fcs, resids = [], []
        for j in range(min(7, len(I[0]))):
            jj = I[0,j]
            if jj<0 or jj>=len(kb_w): continue
            rw, rf = kb_w[jj], kb_f[jj][:pl]
            if len(rf)<pl: continue
            rm, rs = rw.mean(), rw.std()+1e-8
            rp = extractor.predict_median((rw-rm)/rs*ctx_s+ctx_m, pl)
            nn_fcs.append(rp); resids.append(rp - ((rf-rm)/rs*ctx_s+ctx_m))
        if len(nn_fcs)<3: continue
        nn_fcs = np.array(nn_fcs); resids = np.array(resids)
        corrs = nn_fcs - med
        agreement = np.abs(np.mean(np.sign(corrs), axis=0))
        gate = np.clip((agreement-0.4)/0.4, 0, 1)
        bias = np.median(resids, axis=0); mad = np.median(np.abs(resids-bias), axis=0)
        crc_conf = 1.0/(1.0+mad/(np.abs(bias)+1e-8))
        unc_z = (unc-um)/us; u_s = 0.3+0.7/(1.0+np.exp(-unc_z))
        fc = med - 0.35*u_s*crc_conf*bias + 0.25*u_s*gate*np.median(corrs, axis=0)
        rcaf_mse = float(np.mean((fc-gt)**2))
        imp = (base_mse-rcaf_mse)/base_mse*100
        all_rows.append({'dataset':ds,'mean_agreement':float(np.mean(agreement)),
                         'high_agreement_frac':float(np.mean(agreement>=0.6)),
                         'improvement':imp,'base_mse':base_mse,'mean_gate':float(np.mean(gate))})

df = pd.DataFrame(all_rows)
df.to_csv('../results/gate_analysis.csv', index=False)
print(f"\n{'='*60}\nCONSENSUS GATE ANALYSIS\n{'='*60}")
for ds in ['ETTm2','Weather','ETTh1','ECL']:
    sub = df[df.dataset==ds]
    if len(sub)<10: continue
    med_agr = sub.mean_agreement.median()
    high, low = sub[sub.mean_agreement>=med_agr], sub[sub.mean_agreement<med_agr]
    r = np.corrcoef(sub.mean_agreement, sub.improvement)[0,1]
    print(f"\n  {ds} (n={len(sub)}): r(agreement, improvement)={r:+.3f}")
    print(f"    High-agr: avg imp={high.improvement.mean():+.2f}%, win rate={100*(high.improvement>0).mean():.0f}%")
    print(f"    Low-agr:  avg imp={low.improvement.mean():+.2f}%, win rate={100*(low.improvement>0).mean():.0f}%")
    print(f"    Gap: {high.improvement.mean()-low.improvement.mean():+.2f}pp")
