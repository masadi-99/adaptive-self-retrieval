"""
Final evaluation: RCAF (Retrieval-Calibrated Adaptive Forecasting).
CRC + CGPF + uncertainty scaling across all datasets and horizons.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from datasets_loader import load_dataset, DATASET_CONFIGS
from embeddings import ChronosEmbeddingExtractor
from fusion import RCAF, ForecastEnsembleFusion
import faiss

warnings.filterwarnings('ignore')


def build_kb(extractor, data, ws, pl, stride=64):
    """Build KB with windows, futures, and FAISS index."""
    windows, futures, embs = [], [], []
    for start in range(0, len(data) - ws - pl, stride):
        w = data[start:start+ws]
        f = data[start+ws:start+ws+pl]
        if len(f) < pl: continue
        windows.append(w)
        futures.append(f)
        embs.append(extractor.extract_embedding(w))
    index = faiss.IndexFlatL2(extractor.embedding_dim)
    if embs:
        index.add(np.stack(embs).astype(np.float32))
    return windows, futures, index


def eval_methods(extractor, test, ws, pl, kb_windows, kb_futures, faiss_index):
    """Evaluate RCAF + baselines on all test windows."""
    results = {name: [] for name in [
        'baseline', 'dense_bw2', 'rcaf', 'rcaf_crc_only', 'rcaf_cgpf_only',
    ]}

    positions = list(range(0, len(test) - ws - pl + 1, pl))
    total = len(positions)

    # Collect uncertainties for normalization
    all_uncs = []
    for start in positions:
        _, _, unc = extractor.predict_with_uncertainty(test[start:start+ws], pl)
        all_uncs.append(unc)
    unc_mean, unc_std = np.mean(all_uncs), np.std(all_uncs) + 1e-8

    rcaf_full = RCAF(extractor, k=7, alpha=0.35, beta=0.25)
    rcaf_crc = RCAF(extractor, k=7, alpha=0.4, beta=0.0)   # CRC only
    rcaf_cgpf = RCAF(extractor, k=7, alpha=0.0, beta=0.3)  # CGPF only

    for wi, start in enumerate(positions):
        ctx = test[start:start+ws]
        gt = test[start+ws:start+ws+pl]
        ctx_mean, ctx_std = ctx.mean(), ctx.std() + 1e-8

        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        results['baseline'].append(float(np.mean((med - gt)**2)))

        # Dense ensemble (previous best)
        if faiss_index.ntotal > 0:
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = faiss_index.search(emb, 1)
            rw = kb_windows[I[0,0]]
            rw_n = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_std + ctx_mean
            nn_pred = extractor.predict_median(rw_n, pl)
            sim = 1.0 / (1.0 + D[0,0])
            ens = (2.0 * med + sim * nn_pred) / (2.0 + sim)
            results['dense_bw2'].append(float(np.mean((ens - gt)**2)))
        else:
            results['dense_bw2'].append(results['baseline'][-1])

        # RCAF variants
        kwargs = dict(
            current_context=ctx, kb_windows=kb_windows,
            kb_futures=kb_futures, faiss_index=faiss_index,
            unc_mean=unc_mean, unc_std=unc_std,
        )

        for name, rcaf_obj in [('rcaf', rcaf_full), ('rcaf_crc_only', rcaf_crc),
                                ('rcaf_cgpf_only', rcaf_cgpf)]:
            fc = rcaf_obj.fuse(med, unc, None, pl, **kwargs)
            results[name].append(float(np.mean((fc - gt)**2)))

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
            kb_w, kb_f, idx = build_kb(extractor, tv, 512, pl, stride=64)
            print(f"  KB: {len(kb_w)} entries ({time.time()-t0:.0f}s)")

            t0 = time.time()
            metrics = eval_methods(extractor, test, 512, pl, kb_w, kb_f, idx)
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
                print(f"  {name:20s}: MSE={mse:12.4f} ({imp:+6.2f}%){marker}")

            pd.DataFrame(all_rows).to_csv(
                os.path.join(results_dir, 'ablation_final.csv'), index=False)

    # Streaming with RCAF
    for ds in [d for d in ['ETTh1', 'ETTm2', 'Weather'] if d in datasets]:
        print(f"\nStreaming: {ds}")
        train, val, test, _ = load_dataset(ds, data_dir)
        full = np.concatenate([train, val, test])
        ws, pl = 512, 96

        online_windows, online_futures = [], []
        online_embs = []
        index = faiss.IndexFlatL2(extractor.embedding_dim)
        rcaf_model = RCAF(extractor, k=7, alpha=0.35, beta=0.25)

        cmse, wmse, blmse = [], [], []
        unc_history = []

        positions = list(range(ws, len(full) - pl, pl))
        for step, t in enumerate(positions):
            ctx, gt = full[t-ws:t], full[t:t+pl]
            med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
            blmse.append(float(np.mean((med - gt)**2)))
            unc_history.append(unc)

            fc = med.copy()
            if index.ntotal >= 3:
                unc_mean_h = np.mean(unc_history)
                unc_std_h = np.std(unc_history) + 1e-8
                fc = rcaf_model.fuse(
                    med, unc, None, pl,
                    current_context=ctx, kb_windows=online_windows,
                    kb_futures=online_futures, faiss_index=index,
                    unc_mean=unc_mean_h, unc_std=unc_std_h,
                )

            wmse.append(float(np.mean((fc - gt)**2)))
            cmse.append(float(np.mean(wmse)))

            # Add to online KB
            emb = extractor.extract_embedding(ctx)
            online_windows.append(ctx.copy())
            online_futures.append(gt.copy())
            online_embs.append(emb)
            index.add(emb.reshape(1,-1).astype(np.float32))

            if (step+1) % 50 == 0 or (step+1) == len(positions):
                bl_c = float(np.mean(blmse))
                print(f"  {step+1}/{len(positions)}: RCAF={cmse[-1]:.4f} bl={bl_c:.4f}")

        np.save(os.path.join(results_dir, f'streaming_final_{ds}.npy'),
               {'cumulative_mse': cmse, 'window_mses': wmse, 'baseline_mses': blmse})
        imp = (np.mean(blmse) - cmse[-1]) / np.mean(blmse) * 100
        print(f"  {ds}: RCAF={cmse[-1]:.4f} bl={np.mean(blmse):.4f} ({imp:+.2f}%)")


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
