"""Multi-horizon killer experiment: does retrieval vs random pattern hold at H=192,336?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, faiss
from datasets_loader import load_dataset
from embeddings import ChronosEmbeddingExtractor

np.random.seed(42)
extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')

def eval_mode(extractor, test, tv, ws, pl, mode, k=7, beta=0.25):
    if mode == 'retrieved':
        kb_w, kb_e = [], []
        for s in range(0, len(tv)-ws-pl, 128):
            w = tv[s:s+ws]
            kb_w.append(w); kb_e.append(extractor.extract_embedding(w))
        idx = faiss.IndexFlatL2(extractor.embedding_dim)
        idx.add(np.stack(kb_e).astype(np.float32))
    positions = list(range(0, len(test)-ws-pl+1, pl))[:40]
    if not positions: return 0.0
    bm, cm = [], []
    for s in positions:
        ctx = test[s:s+ws]; gt = test[s+ws:s+ws+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std()+1e-8
        med = extractor.predict_median(ctx, pl); bm.append(float(np.mean((med-gt)**2)))
        nn_fc = []
        if mode == 'retrieved':
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = idx.search(emb, k)
            for j in range(min(k, len(I[0]))):
                rw = kb_w[I[0,j]]
                nn_fc.append(extractor.predict_median((rw-rw.mean())/(rw.std()+1e-8)*ctx_s+ctx_m, pl))
        else:
            for _ in range(k):
                nn_fc.append(extractor.predict_median(np.random.randn(ws).astype(np.float32)*ctx_s+ctx_m, pl))
        if len(nn_fc)<3: cm.append(bm[-1]); continue
        nn_fc=np.array(nn_fc); corr=nn_fc-med
        agr=np.abs(np.mean(np.sign(corr),axis=0))
        gate=np.clip((agr-0.4)/0.4,0,1)
        cm.append(float(np.mean(((med+beta*gate*np.median(corr,axis=0))-gt)**2)))
    bl,rc=np.mean(bm),np.mean(cm)
    return (bl-rc)/bl*100 if bl>0 else 0.0

print("MULTI-HORIZON KILLER EXPERIMENT")
print("="*60)
print(f"{'DS':<10} {'H':>5} {'Retrieved':>10} {'Random':>10} {'Gap':>8}")
print("-"*50)

rows = []
for ds in ['ETTm2','ETTh1','Weather','ETTm1','ETTh2']:
    train, val, test, config = load_dataset(ds, '../data')
    tv = np.concatenate([train, val])
    for pl in [96, 192, 336]:
        r = eval_mode(extractor, test, tv, 512, pl, 'retrieved')
        n = eval_mode(extractor, test, tv, 512, pl, 'random')
        gap = r - n
        winner = 'RET' if gap > 1 else 'RND' if gap < -1 else 'TIE'
        print(f"{ds:<10} {pl:>5} {r:>+9.1f}% {n:>+9.1f}% {gap:>+7.1f}pp {winner}")
        rows.append({'dataset':ds,'pred_len':pl,'retrieved':r,'random':n,'gap':gap})

import pandas as pd
pd.DataFrame(rows).to_csv('../results/killer_multihorizon.csv', index=False)
print("\nSaved results/killer_multihorizon.csv")
