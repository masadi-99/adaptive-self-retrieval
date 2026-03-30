"""S2: Empirical coverage analysis of CRC-calibrated prediction intervals."""
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

rows = []
for ds in ['ETTm2','Weather','ETTh1','ECL']:
    train, val, test, _ = load_dataset(ds, '../data')
    tv = np.concatenate([train, val])
    for pl in [96, 336]:
        print(f"\n  Coverage: {ds} H={pl}")
        kb_w, kb_f, idx = build_kb(tv, 512, pl)
        positions = list(range(0, len(test)-512-pl+1, pl))
        base_in, rcaf_in, total_ts = 0, 0, 0
        for start in positions:
            ctx, gt = test[start:start+512], test[start+512:start+512+pl]
            ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
            _, quants, _ = extractor.predict_with_uncertainty(ctx, pl)
            bq10, bq50, bq90 = quants[0], quants[1], quants[2]
            base_in += np.sum((gt >= bq10) & (gt <= bq90)); total_ts += pl
            if idx.ntotal < 5: rcaf_in += np.sum((gt >= bq10) & (gt <= bq90)); continue
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = idx.search(emb, 7)
            rq = {i: [] for i in range(3)}
            for j in range(min(7, len(I[0]))):
                jj = I[0,j]
                if jj<0 or jj>=len(kb_w): continue
                rw, rf = kb_w[jj], kb_f[jj][:pl]
                if len(rf)<pl: continue
                rm, rs = rw.mean(), rw.std()+1e-8
                _, nq, _ = extractor.predict_with_uncertainty((rw-rm)/rs*ctx_s+ctx_m, pl)
                rf_a = (rf-rm)/rs*ctx_s+ctx_m
                for qi in range(3): rq[qi].append(nq[qi] - rf_a)
            if len(rq[0])<2: rcaf_in += np.sum((gt >= bq10) & (gt <= bq90)); continue
            def cal(bq, rl, alpha=0.35):
                r=np.array(rl); b=np.median(r,axis=0)
                m=np.median(np.abs(r-b),axis=0); c=1.0/(1.0+m/(np.abs(b)+1e-8))
                return bq - alpha*c*b
            cq50 = cal(bq50, rq[1]); cq10 = np.minimum(cal(bq10, rq[0]), cq50)
            cq90 = np.maximum(cal(bq90, rq[2]), cq50)
            rcaf_in += np.sum((gt >= cq10) & (gt <= cq90))
        bc = base_in/total_ts*100; rc = rcaf_in/total_ts*100
        print(f"    80% PI: base={bc:.1f}%  RCAF={rc:.1f}%  (target=80%)")
        rows.append({'dataset':ds,'pred_len':pl,'base_coverage':bc,'rcaf_coverage':rc,
                     'target':80.0,'n_windows':len(positions)})

pd.DataFrame(rows).to_csv('../results/coverage_results.csv', index=False)
print("\nSummary:")
for _, r in pd.DataFrame(rows).iterrows():
    bg, rg = abs(r.base_coverage-80), abs(r.rcaf_coverage-80)
    print(f"  {r.dataset} H={int(r.pred_len)}: {r.base_coverage:.1f}% → {r.rcaf_coverage:.1f}% "
          f"[{'RCAF closer' if rg<bg else 'Base closer'}]")
