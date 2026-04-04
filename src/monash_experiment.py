"""
Spotlight experiment: validate spectral classifier on diverse datasets.
Predicts whether retrieval or random augmentation helps more, then measures actual.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import faiss
from scipy import stats
from embeddings import ChronosEmbeddingExtractor
from numpy.fft import rfft, rfftfreq

np.random.seed(42)
extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', 'cuda')


def spectral_concentration(series, n=4096):
    """Fraction of spectral power in top-3 frequencies. High = periodic."""
    x = series[-min(len(series), n):]
    x = x - x.mean()
    power = np.abs(rfft(x))**2
    if power[1:].sum() < 1e-10:
        return 0.0
    top3 = np.sort(power[1:])[-3:].sum()
    return float(top3 / power[1:].sum())


def eval_mode(extractor, test, train_data, ws, pl, mode, k=7, beta=0.25):
    """Evaluate a TTA mode. Returns improvement %."""
    if mode == 'retrieved':
        kb_w, kb_e = [], []
        for s in range(0, len(train_data)-ws-pl, 128):
            w = train_data[s:s+ws]
            kb_w.append(w); kb_e.append(extractor.extract_embedding(w))
        if not kb_e:
            return 0.0
        idx = faiss.IndexFlatL2(extractor.embedding_dim)
        idx.add(np.stack(kb_e).astype(np.float32))

    positions = list(range(0, len(test)-ws-pl+1, pl))[:40]
    if not positions:
        return 0.0

    bm, cm = [], []
    for s in positions:
        ctx = test[s:s+ws]; gt = test[s+ws:s+ws+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std() + 1e-8
        med = extractor.predict_median(ctx, pl)
        bm.append(float(np.mean((med-gt)**2)))

        nn_fc = []
        if mode == 'retrieved':
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = idx.search(emb, k)
            for j in range(min(k, len(I[0]))):
                rw = kb_w[I[0,j]]
                nn_fc.append(extractor.predict_median(
                    (rw-rw.mean())/(rw.std()+1e-8)*ctx_s+ctx_m, pl))
        else:  # random
            for _ in range(k):
                nn_fc.append(extractor.predict_median(
                    np.random.randn(ws).astype(np.float32)*ctx_s+ctx_m, pl))

        if len(nn_fc) < 3:
            cm.append(bm[-1]); continue
        nn_fc = np.array(nn_fc)
        corr = nn_fc - med
        agr = np.abs(np.mean(np.sign(corr), axis=0))
        gate = np.clip((agr-0.4)/0.4, 0, 1)
        corrected = med + beta * gate * np.median(corr, axis=0)
        cm.append(float(np.mean((corrected-gt)**2)))

    bl, rc = np.mean(bm), np.mean(cm)
    return (bl-rc)/bl*100 if bl > 0 else 0.0


# Datasets to test
from datasets_loader import load_dataset, DATASET_CONFIGS

datasets_to_test = list(DATASET_CONFIGS.keys())
# Remove ILI (too short)
datasets_to_test = [d for d in datasets_to_test if d != 'ILI']

print("SPOTLIGHT: Spectral Classifier for Retrieval vs Random")
print("="*75)
print(f"{'Dataset':<12} {'SpectConc':>10} {'Retrieved':>10} {'Random':>10} {'Winner':>10} {'Predicted':>10} {'Correct':>8}")
print("-"*75)

rows = []
for ds in datasets_to_test:
    try:
        train, val, test, config = load_dataset(ds, '../data')
    except Exception as e:
        print(f"{ds}: SKIP ({e})")
        continue

    tv = np.concatenate([train, val])
    pl = config['prediction_lengths'][0]  # shortest horizon

    # Spectral concentration
    sc = spectral_concentration(train)

    # Evaluate both modes
    imp_ret = eval_mode(extractor, test, tv, 512, pl, 'retrieved', k=7)
    imp_rnd = eval_mode(extractor, test, tv, 512, pl, 'random', k=7)

    actual_winner = 'retrieval' if imp_ret > imp_rnd else 'random'
    # Prediction: low spectral concentration → retrieval helps
    predicted = 'retrieval' if sc < 0.3 else 'random'
    correct = actual_winner == predicted

    print(f"{ds:<12} {sc:>10.3f} {imp_ret:>+9.1f}% {imp_rnd:>+9.1f}% {actual_winner:>10} {predicted:>10} {'YES' if correct else 'NO':>8}")

    rows.append({
        'dataset': ds, 'spectral_concentration': sc,
        'retrieved_imp': imp_ret, 'random_imp': imp_rnd,
        'actual_winner': actual_winner, 'predicted': predicted,
        'correct': correct,
    })

df = pd.DataFrame(rows)
df.to_csv('../results/monash_spectral.csv', index=False)

accuracy = df.correct.mean() * 100
print(f"\nClassifier accuracy: {accuracy:.0f}% ({df.correct.sum()}/{len(df)})")
print(f"Threshold: spectral_concentration < 0.3 → retrieval; >= 0.3 → random")
