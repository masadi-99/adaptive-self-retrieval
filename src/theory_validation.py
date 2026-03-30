"""Theory validation: does CRC bias estimate predict actual forecast error?"""
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

all_rows = []
for ds in ['ETTm2', 'Weather', 'ETTh1']:
    train, val, test, config = load_dataset(ds, '../data')
    tv = np.concatenate([train, val])
    pl = 96
    print(f"\n  Theory: {ds} H={pl}")
    kb_w, kb_f, idx = build_kb(tv, 512, pl)
    positions = list(range(0, len(test)-512-pl+1, pl))
    uncs = [extractor.predict_with_uncertainty(test[s:s+512], pl)[2] for s in positions]
    um, us = np.mean(uncs), np.std(uncs)+1e-8
    rcaf_model = RCAF(extractor, k=7, alpha=0.35, beta=0.25)

    for wi, start in enumerate(positions):
        ctx, gt = test[start:start+512], test[start+512:start+512+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        actual_error = med - gt
        if idx.ntotal < 5: continue
        emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
        D, I = idx.search(emb, 7)
        residuals = []
        for j in range(min(7, len(I[0]))):
            jj = I[0,j]
            if jj < 0 or jj >= len(kb_w): continue
            rw, rf = kb_w[jj], kb_f[jj][:pl]
            if len(rf) < pl: continue
            rm, rs = rw.mean(), rw.std()+1e-8
            rp = extractor.predict_median((rw-rm)/rs*ctx_s+ctx_m, pl)
            residuals.append(rp - ((rf-rm)/rs*ctx_s+ctx_m))
        if len(residuals) < 3: continue
        bias_est = np.median(np.array(residuals), axis=0)
        r = np.corrcoef(bias_est, actual_error)[0, 1]
        if np.isnan(r): continue
        base_mse = float(np.mean(actual_error**2))
        fc = rcaf_model.fuse(med, unc, None, pl, current_context=ctx,
                             kb_windows=kb_w, kb_futures=kb_f, faiss_index=idx,
                             unc_mean=um, unc_std=us)
        rcaf_mse = float(np.mean((fc-gt)**2))
        imp = (base_mse-rcaf_mse)/base_mse*100
        all_rows.append({'dataset':ds, 'window':wi, 'bias_error_corr':r,
                         'base_mse':base_mse, 'improvement':imp, 'uncertainty':unc})
        if (wi+1) % 30 == 0: print(f"    {wi+1}/{len(positions)}")

df = pd.DataFrame(all_rows)
df.to_csv('../results/theory_validation.csv', index=False)
print(f"\n{'='*60}")
print("THEORY VALIDATION RESULTS")
print(f"{'='*60}")
for ds in ['ETTm2', 'Weather', 'ETTh1']:
    sub = df[df.dataset == ds]
    if len(sub) < 5: continue
    r1 = np.corrcoef(sub.bias_error_corr, sub.improvement)[0, 1]
    r2 = np.corrcoef(sub.uncertainty, sub.improvement)[0, 1]
    print(f"\n  {ds} (n={len(sub)} windows):")
    print(f"    r(bias_error_corr, improvement) = {r1:+.3f}  <- THEORY PREDICTION")
    print(f"    r(uncertainty, improvement)      = {r2:+.3f}")
    print(f"    Mean bias_error_corr = {sub.bias_error_corr.mean():.3f}")
