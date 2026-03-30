"""S1: Functional vs native embedding comparison for RCAF."""
import os, sys, numpy as np, pandas as pd, faiss, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor

extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')

class FunctionalChronosAdapter:
    def __init__(self, ext, offsets=(0,32,64,128), emb_horizon=64):
        self.extractor = ext; self.offsets = offsets; self.emb_horizon = emb_horizon
        self.embedding_dim = len(offsets) * emb_horizon
    def extract_embedding(self, w):
        parts = []
        for off in self.offsets:
            sub = w[off:] if len(w[off:]) >= 128 else w
            p = self.extractor.predict_median(sub, self.emb_horizon)
            parts.append((p - p.mean()) / (p.std() + 1e-8))
        return np.concatenate(parts).astype(np.float32)
    def predict_median(self, c, pl): return self.extractor.predict_median(c, pl)
    def predict_with_uncertainty(self, c, pl): return self.extractor.predict_with_uncertainty(c, pl)

func_adapter = FunctionalChronosAdapter(extractor)

def build_kb(adapter, data, ws, pl, stride=64):
    w, f, e = [], [], []
    for s in range(0, len(data)-ws-pl, stride):
        ww, ff = data[s:s+ws], data[s+ws:s+ws+pl]
        if len(ff)<pl: continue
        w.append(ww); f.append(ff); e.append(adapter.extract_embedding(ww))
    idx = faiss.IndexFlatL2(adapter.embedding_dim)
    if e: idx.add(np.stack(e).astype(np.float32))
    return w, f, idx

def eval_rcaf(adapter, test, ws, pl, kb_w, kb_f, idx, k=7, alpha=0.35, beta=0.25):
    positions = list(range(0, len(test)-ws-pl+1, pl))
    uncs = [adapter.predict_with_uncertainty(test[s:s+ws], pl)[2] for s in positions]
    um, us = np.mean(uncs), np.std(uncs)+1e-8
    bm, rm = [], []
    for s in positions:
        ctx, gt = test[s:s+ws], test[s+ws:s+ws+pl]
        ctx_s = ctx.std()+1e-8; ctx_m = ctx.mean()
        med, _, unc = adapter.predict_with_uncertainty(ctx, pl)
        bm.append(float(np.mean((med-gt)**2)))
        if idx.ntotal < 3: rm.append(bm[-1]); continue
        emb = adapter.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
        D, I = idx.search(emb, k)
        nn_fcs, resids = [], []
        for j in range(min(k, len(I[0]))):
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
        fc=med-alpha*u_s*crc_conf*bias+beta*u_s*gate*np.median(corrs,axis=0)
        rm.append(float(np.mean((fc-gt)**2)))
    return np.mean(bm), np.mean(rm)

rows = []
for ds in ['ETTh1','ETTm1','ETTm2','Weather']:
    train, val, test, _ = load_dataset(ds, '../data')
    tv = np.concatenate([train, val]); pl = 96
    print(f"\n{ds} H={pl}")
    kb_wn, kb_fn, idx_n = build_kb(extractor, tv, 512, pl)
    bl_n, rc_n = eval_rcaf(extractor, test, 512, pl, kb_wn, kb_fn, idx_n)
    imp_n = (bl_n-rc_n)/bl_n*100
    print(f"  Native  (512d): {imp_n:+.2f}%")
    rows.append({'dataset':ds,'embedding':'native_512d','improvement':imp_n})
    kb_wf, kb_ff, idx_f = build_kb(func_adapter, tv, 512, pl)
    bl_f, rc_f = eval_rcaf(func_adapter, test, 512, pl, kb_wf, kb_ff, idx_f)
    imp_f = (bl_f-rc_f)/bl_f*100
    print(f"  Funct.  (256d): {imp_f:+.2f}%")
    rows.append({'dataset':ds,'embedding':'functional_256d','improvement':imp_f})
    print(f"  Delta: {imp_f-imp_n:+.2f}pp {'(functional wins)' if imp_f>imp_n else '(native wins)'}")
pd.DataFrame(rows).to_csv('../results/functional_vs_native.csv', index=False)
print("\nSaved results/functional_vs_native.csv")
