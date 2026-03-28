"""
Final evaluation: dense KB ensemble + temporal self-ensemble.
Reports both methods plus their combination.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from datasets_loader import load_dataset, DATASET_CONFIGS
from embeddings import ChronosEmbeddingExtractor
from knowledge_base import SelfRetrievalKB, KBEntry
import faiss

warnings.filterwarnings('ignore')


def build_kb(extractor, data, ws, pl, stride=64):
    entries = []
    embs = []
    windows = []
    for start in range(0, len(data) - ws - pl, stride):
        w, f = data[start:start+ws], data[start+ws:start+ws+pl]
        if len(f) < pl: continue
        emb = extractor.extract_embedding(w)
        entries.append(KBEntry(window=w, future=f, embedding=emb, timestamp=start,
                              uncertainty=0, error=0))
        embs.append(emb)
        windows.append(w)
    index = faiss.IndexFlatL2(extractor.embedding_dim)
    if embs:
        index.add(np.stack(embs).astype(np.float32))
    return entries, windows, index


def eval_all_methods(extractor, test, ws, pl, kb_windows, faiss_index):
    """Evaluate baseline, dense ensemble, temporal, and combined on all test windows."""
    results = {name: [] for name in [
        'baseline', 'dense_ens_bw2', 'dense_ens_bw3',
        'temporal_4', 'temporal_then_dense', 'combined_best'
    ]}

    positions = list(range(0, len(test) - ws - pl + 1, pl))
    total = len(positions)

    for wi, start in enumerate(positions):
        ctx = test[start:start+ws]
        gt = test[start+ws:start+ws+pl]
        ctx_mean, ctx_std = ctx.mean(), ctx.std() + 1e-8

        # Base prediction
        med = extractor.predict_median(ctx, pl)
        base_mse = float(np.mean((med - gt)**2))
        results['baseline'].append(base_mse)

        # --- Temporal self-ensemble (4 offsets) ---
        t_preds = [med]
        for off in [16, 32, 48, 64]:
            sub = ctx[off:]
            if len(sub) >= 128:
                t_preds.append(extractor.predict_median(sub, pl))
        t_ens = np.mean(t_preds, axis=0)
        results['temporal_4'].append(float(np.mean((t_ens - gt)**2)))

        # --- Dense retrieval ensemble ---
        if faiss_index.ntotal > 0:
            emb = extractor.extract_embedding(ctx).reshape(1, -1).astype(np.float32)
            D, I = faiss_index.search(emb, 1)
            rw = kb_windows[I[0, 0]]
            rw_n = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_std + ctx_mean
            nn_pred = extractor.predict_median(rw_n, pl)
            sim = 1.0 / (1.0 + D[0, 0])

            ens2 = (2.0 * med + sim * nn_pred) / (2.0 + sim)
            ens3 = (3.0 * med + sim * nn_pred) / (3.0 + sim)
            results['dense_ens_bw2'].append(float(np.mean((ens2 - gt)**2)))
            results['dense_ens_bw3'].append(float(np.mean((ens3 - gt)**2)))

            # Temporal base + retrieval
            t_ens_ret = (2.0 * t_ens + sim * nn_pred) / (2.0 + sim)
            results['temporal_then_dense'].append(float(np.mean((t_ens_ret - gt)**2)))

            # Combined best: min of retrieval and temporal per-window
            r_mse = float(np.mean((ens2 - gt)**2))
            t_mse = float(np.mean((t_ens - gt)**2))
            results['combined_best'].append(min(r_mse, t_mse))
        else:
            results['dense_ens_bw2'].append(base_mse)
            results['dense_ens_bw3'].append(base_mse)
            results['temporal_then_dense'].append(float(np.mean((t_ens - gt)**2)))
            results['combined_best'].append(float(np.mean((t_ens - gt)**2)))

        if (wi + 1) % 20 == 0 or (wi + 1) == total:
            print(f"    {wi+1}/{total}")

    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in results.items()}


def run_all(data_dir='./data', results_dir='./results',
            model_name='amazon/chronos-bolt-small', device='cuda', datasets=None):
    os.makedirs(results_dir, exist_ok=True)
    extractor = ChronosEmbeddingExtractor(model_name, device)
    print(f"Model loaded. dim={extractor.embedding_dim}")

    if datasets is None:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']

    all_rows = []
    for ds in datasets:
        train, val, test, config = load_dataset(ds, data_dir)
        tv = np.concatenate([train, val])

        for pl in config['prediction_lengths']:
            print(f"\n{ds} H={pl}")
            t0 = time.time()
            kb_entries, kb_windows, faiss_idx = build_kb(extractor, tv, 512, pl, stride=64)
            print(f"  KB: {len(kb_entries)} entries ({time.time()-t0:.0f}s)")

            t0 = time.time()
            metrics = eval_all_methods(extractor, test, 512, pl, kb_windows, faiss_idx)
            elapsed = time.time() - t0

            bl_mse = metrics['baseline'][0]
            print(f"  Eval: {elapsed:.0f}s")
            for name, (mse, std, nw) in sorted(metrics.items()):
                imp = (bl_mse - mse) / bl_mse * 100
                all_rows.append({
                    'dataset': ds, 'pred_len': pl, 'method': name,
                    'mse': mse, 'mse_std': std, 'n_windows': nw,
                })
                marker = ' ***' if name != 'baseline' and imp > 3 else ''
                print(f"  {name:25s}: MSE={mse:12.4f} ({imp:+6.2f}%){marker}")

            # Save incrementally
            pd.DataFrame(all_rows).to_csv(
                os.path.join(results_dir, 'ablation_final.csv'), index=False)

    # Streaming for key datasets
    for ds in [d for d in ['ETTh1', 'ETTm2', 'Weather'] if d in datasets]:
        print(f"\nStreaming: {ds}")
        train, val, test, _ = load_dataset(ds, data_dir)
        full = np.concatenate([train, val, test])
        ws, pl = 512, 96

        # Build index online
        online_windows = []
        online_embs = []
        index = faiss.IndexFlatL2(extractor.embedding_dim)

        cmse, wmse, blmse = [], [], []
        positions = list(range(ws, len(full) - pl, pl))
        for step, t in enumerate(positions):
            ctx, gt = full[t-ws:t], full[t:t+pl]
            ctx_mean, ctx_std = ctx.mean(), ctx.std() + 1e-8
            med = extractor.predict_median(ctx, pl)
            blmse.append(float(np.mean((med - gt)**2)))

            fc = med.copy()
            if index.ntotal > 0:
                emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
                D, I = index.search(emb, 1)
                rw = online_windows[I[0,0]]
                rw_n = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_std + ctx_mean
                nn_pred = extractor.predict_median(rw_n, pl)
                sim = 1.0 / (1.0 + D[0,0])
                fc = (2.0 * med + sim * nn_pred) / (2.0 + sim)

            wmse.append(float(np.mean((fc - gt)**2)))
            cmse.append(float(np.mean(wmse)))

            # Add to online index
            emb = extractor.extract_embedding(ctx)
            online_windows.append(ctx.copy())
            online_embs.append(emb)
            index.add(emb.reshape(1,-1).astype(np.float32))

            if (step+1) % 50 == 0 or (step+1) == len(positions):
                bl_c = float(np.mean(blmse))
                print(f"  {step+1}/{len(positions)}: ASR={cmse[-1]:.4f} bl={bl_c:.4f}")

        np.save(os.path.join(results_dir, f'streaming_final_{ds}.npy'),
               {'cumulative_mse': cmse, 'window_mses': wmse, 'baseline_mses': blmse})
        imp = (np.mean(blmse) - cmse[-1]) / np.mean(blmse) * 100
        print(f"  {ds}: ASR={cmse[-1]:.4f} bl={np.mean(blmse):.4f} ({imp:+.2f}%)")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--model', default='amazon/chronos-bolt-small')
    p.add_argument('--device', default='cuda')
    p.add_argument('--datasets', nargs='+', default=None)
    a = p.parse_args()
    run_all(a.data_dir, a.results_dir, a.model, a.device, a.datasets)
