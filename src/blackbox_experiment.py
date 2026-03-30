"""S5: RCAF as pure black-box wrapper — only predict() access, no model internals."""
import os, sys, numpy as np, pandas as pd, faiss, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor

_model = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')

class BlackBoxAdapter:
    def __init__(self, predict_fn, offsets=(0,32,64,128), emb_horizon=64):
        self.predict_fn = predict_fn; self.offsets = offsets; self.emb_horizon = emb_horizon
        self.embedding_dim = len(offsets) * emb_horizon
    def extract_embedding(self, w):
        parts = []
        for off in self.offsets:
            sub = w[off:] if len(w[off:])>=128 else w
            p = self.predict_fn(sub, self.emb_horizon)
            parts.append((p-p.mean())/(p.std()+1e-8))
        return np.concatenate(parts).astype(np.float32)
    def predict_median(self, c, pl): return self.predict_fn(c, pl)
    def predict_with_uncertainty(self, c, pl):
        preds = [self.predict_fn(c[off:], pl) for off in [0,16,32,48] if len(c[off:])>=128]
        if len(preds)<2:
            m = self.predict_fn(c, pl); return m, np.stack([m]*3), 0.0
        preds = np.array(preds); med = np.median(preds, axis=0)
        return med, np.stack([med]*3), float(np.mean(np.std(preds, axis=0)))

adapter = BlackBoxAdapter(predict_fn=_model.predict_median)

def build_kb(data, ws, pl, stride=64):
    w, f, e = [], [], []
    for s in range(0, len(data)-ws-pl, stride):
        ww, ff = data[s:s+ws], data[s+ws:s+ws+pl]
        if len(ff)<pl: continue
        w.append(ww); f.append(ff); e.append(adapter.extract_embedding(ww))
    idx = faiss.IndexFlatL2(adapter.embedding_dim)
    if e: idx.add(np.stack(e).astype(np.float32))
    return w, f, idx

rows = []
for ds in ['ETTh1','ETTm1','ETTm2','Weather']:
    train, val, test, _ = load_dataset(ds, '../data')
    tv = np.concatenate([train, val]); pl = 96
    print(f"\n  Black-box: {ds} H={pl}")
    kb_w, kb_f, idx = build_kb(tv, 512, pl)
    positions = list(range(0, len(test)-512-pl+1, pl))
    uncs = [adapter.predict_with_uncertainty(test[s:s+512], pl)[2] for s in positions]
    um, us = np.mean(uncs), np.std(uncs)+1e-8
    bm, rm = [], []
    for wi, s in enumerate(positions):
        ctx, gt = test[s:s+512], test[s+512:s+512+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
        med, _, unc = adapter.predict_with_uncertainty(ctx, pl)
        bm.append(float(np.mean((med-gt)**2)))
        if idx.ntotal<3: rm.append(bm[-1]); continue
        emb = adapter.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
        D, I = idx.search(emb, 7)
        nn_fcs, resids = [], []
        for j in range(min(7, len(I[0]))):
            jj = I[0,j]
            if jj<0 or jj>=len(kb_w): continue
            rw, rf = kb_w[jj], kb_f[jj][:pl]
            if len(rf)<pl: continue
            rm2, rs = rw.mean(), rw.std()+1e-8
            rp = adapter.predict_median((rw-rm2)/rs*ctx_s+ctx_m, pl)
            nn_fcs.append(rp); resids.append(rp - ((rf-rm2)/rs*ctx_s+ctx_m))
        if len(nn_fcs)<3: rm.append(bm[-1]); continue
        nn_fcs=np.array(nn_fcs); resids=np.array(resids)
        bias=np.median(resids,axis=0); mad=np.median(np.abs(resids-bias),axis=0)
        crc_conf=1.0/(1.0+mad/(np.abs(bias)+1e-8))
        corrs=nn_fcs-med; gate=np.clip((np.abs(np.mean(np.sign(corrs),axis=0))-0.4)/0.4,0,1)
        unc_z=(unc-um)/us; u_s=0.3+0.7/(1.0+np.exp(-unc_z))
        fc=med-0.35*u_s*crc_conf*bias+0.25*u_s*gate*np.median(corrs,axis=0)
        rm.append(float(np.mean((fc-gt)**2)))
        if (wi+1)%30==0: print(f"    {wi+1}/{len(positions)}")
    bl, rc = np.mean(bm), np.mean(rm); imp = (bl-rc)/bl*100
    print(f"    baseline={bl:.4f}  blackbox RCAF={rc:.4f}  ({imp:+.2f}%)")
    rows.append({'dataset':ds,'mode':'blackbox','base_mse':bl,'rcaf_mse':rc,'improvement':imp})

pd.DataFrame(rows).to_csv('../results/blackbox_results.csv', index=False)
native = {'ETTh1':6.8,'ETTm1':4.2,'ETTm2':4.6,'Weather':8.3}
print(f"\n{'='*60}\nBLACK-BOX vs NATIVE RCAF\n{'='*60}")
print(f"{'DS':10s} {'Native':>10s} {'BlackBox':>10s} {'Gap':>8s}")
for ds in ['ETTh1','ETTm1','ETTm2','Weather']:
    n = native[ds]
    b = pd.DataFrame(rows); b = b[b.dataset==ds].improvement.values[0]
    print(f"{ds:10s} {n:>+9.1f}% {b:>+9.1f}% {b-n:>+7.1f}pp")
