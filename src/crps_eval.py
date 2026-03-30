"""CRPS evaluation: probabilistic forecast calibration with CRC."""
import os, sys, numpy as np, pandas as pd, faiss, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor

extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')

def crps_quantile(q_levels, q_forecasts, actual):
    score = 0.0
    for i, tau in enumerate(q_levels):
        err = actual - q_forecasts[i]
        score += np.mean(np.where(err >= 0, 2 * tau * err, -2 * (1 - tau) * err))
    return score / len(q_levels)

def build_kb(data, ws, pl, stride=64):
    w, f, e = [], [], []
    for s in range(0, len(data)-ws-pl, stride):
        ww, ff = data[s:s+ws], data[s+ws:s+ws+pl]
        if len(ff)<pl: continue
        w.append(ww); f.append(ff); e.append(extractor.extract_embedding(ww))
    idx = faiss.IndexFlatL2(extractor.embedding_dim)
    if e: idx.add(np.stack(e).astype(np.float32))
    return w, f, idx

q_levels = [0.1, 0.5, 0.9]
rows = []
for ds in ['ETTm2', 'Weather', 'ETTh1', 'ECL']:
    train, val, test, config = load_dataset(ds, '../data')
    tv = np.concatenate([train, val])
    for pl in [96, 336]:
        print(f"\n  CRPS: {ds} H={pl}")
        kb_w, kb_f, idx = build_kb(tv, 512, pl)
        positions = list(range(0, len(test)-512-pl+1, pl))
        base_crps, rcaf_crps = [], []
        for wi, start in enumerate(positions):
            ctx, gt = test[start:start+512], test[start+512:start+512+pl]
            ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
            med, quants, unc = extractor.predict_with_uncertainty(ctx, pl)
            base_crps.append(crps_quantile(q_levels, quants, gt))
            if idx.ntotal < 5:
                rcaf_crps.append(base_crps[-1]); continue
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = idx.search(emb, 7)
            rq = {i: [] for i in range(3)}
            for j in range(min(7, len(I[0]))):
                jj = I[0,j]
                if jj < 0 or jj >= len(kb_w): continue
                rw, rf = kb_w[jj], kb_f[jj][:pl]
                if len(rf) < pl: continue
                rm, rs = rw.mean(), rw.std()+1e-8
                rw_n = (rw-rm)/rs*ctx_s+ctx_m
                rf_a = (rf-rm)/rs*ctx_s+ctx_m
                _, nq, _ = extractor.predict_with_uncertainty(rw_n, pl)
                for qi in range(3):
                    rq[qi].append(nq[qi] - rf_a)
            if len(rq[0]) < 2:
                rcaf_crps.append(base_crps[-1]); continue
            def cal(bq, rl, alpha=0.35):
                r = np.array(rl); b = np.median(r, axis=0)
                m = np.median(np.abs(r-b), axis=0)
                c = 1.0/(1.0+m/(np.abs(b)+1e-8))
                return bq - alpha*c*b
            cq = [cal(quants[i], rq[i]) for i in range(3)]
            cq[0] = np.minimum(cq[0], cq[1])
            cq[2] = np.maximum(cq[2], cq[1])
            rcaf_crps.append(crps_quantile(q_levels, np.stack(cq), gt))
            if (wi+1) % 20 == 0: print(f"    {wi+1}/{len(positions)}")
        bc, rc = np.mean(base_crps), np.mean(rcaf_crps)
        imp = (bc-rc)/bc*100
        print(f"    Base CRPS={bc:.4f}  RCAF CRPS={rc:.4f}  ({imp:+.2f}%)")
        rows.append({'dataset':ds, 'pred_len':pl, 'base_crps':bc,
                     'rcaf_crps':rc, 'improvement':imp, 'n_windows':len(positions)})

pd.DataFrame(rows).to_csv('../results/crps_results.csv', index=False)
print("\nSaved results/crps_results.csv")
