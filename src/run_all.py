"""
Canonical evaluation script. Produces ALL results in one reproducible run.
Outputs: ablation_final.csv, paired_ttests.txt
Uses a single RCAF implementation from fusion.py (no inline reimplementation).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datasets_loader import load_dataset, DATASET_CONFIGS
from embeddings import ChronosEmbeddingExtractor
from fusion import RCAF
import faiss

warnings.filterwarnings('ignore')
np.random.seed(42)


def build_kb(extractor, data, ws, pl, stride=64):
    windows, futures, embs = [], [], []
    for start in range(0, len(data) - ws - pl, stride):
        w, f = data[start:start+ws], data[start+ws:start+ws+pl]
        if len(f) < pl: continue
        windows.append(w); futures.append(f)
        embs.append(extractor.extract_embedding(w))
    index = faiss.IndexFlatL2(extractor.embedding_dim)
    if embs:
        index.add(np.stack(embs).astype(np.float32))
    return windows, futures, index


def eval_perwindow(extractor, test, ws, pl, kb_w, kb_f, faiss_index, k=7,
                   alpha=0.35, beta=0.25):
    """Evaluate RCAF on all test windows. Returns per-window base and RCAF MSEs."""
    positions = list(range(0, len(test) - ws - pl + 1, pl))
    if not positions:
        return np.array([]), np.array([]), np.array([])

    # Collect uncertainties for normalization
    all_uncs = []
    for start in positions:
        _, _, unc = extractor.predict_with_uncertainty(test[start:start+ws], pl)
        all_uncs.append(unc)
    unc_mean, unc_std = np.mean(all_uncs), np.std(all_uncs) + 1e-8

    rcaf_model = RCAF(extractor, k=k, alpha=alpha, beta=beta)

    base_mses, rcaf_mses, dense_mses = [], [], []
    for wi, start in enumerate(positions):
        ctx = test[start:start+ws]
        gt = test[start+ws:start+ws+pl]
        ctx_m, ctx_s = ctx.mean(), ctx.std() + 1e-8

        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        base_mses.append(float(np.mean((med - gt)**2)))

        # RCAF (k=7, CRC+CGPF+uncertainty scaling)
        fc = rcaf_model.fuse(
            med, unc, None, pl,
            current_context=ctx,
            kb_windows=kb_w, kb_futures=kb_f, faiss_index=faiss_index,
            unc_mean=unc_mean, unc_std=unc_std,
        )
        rcaf_mses.append(float(np.mean((fc - gt)**2)))

        # Dense ensemble baseline (k=1, for comparison)
        if faiss_index.ntotal > 0:
            emb = extractor.extract_embedding(ctx).reshape(1,-1).astype(np.float32)
            D, I = faiss_index.search(emb, 1)
            rw = kb_w[I[0,0]]
            rw_n = (rw - rw.mean()) / (rw.std() + 1e-8) * ctx_s + ctx_m
            nn_pred = extractor.predict_median(rw_n, pl)
            sim = 1.0 / (1.0 + D[0,0])
            ens = (2.0 * med + sim * nn_pred) / (2.0 + sim)
            dense_mses.append(float(np.mean((ens - gt)**2)))
        else:
            dense_mses.append(base_mses[-1])

        if (wi + 1) % 20 == 0 or (wi + 1) == len(positions):
            print(f"    {wi+1}/{len(positions)}")

    return np.array(base_mses), np.array(rcaf_mses), np.array(dense_mses)


def run_all(data_dir='./data', results_dir='./results', device='cuda',
            datasets=None):
    os.makedirs(results_dir, exist_ok=True)
    extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', device)
    print(f"Model loaded. dim={extractor.embedding_dim}")

    if datasets is None:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']

    all_ablation = []
    all_ttests = []

    for ds in datasets:
        train, val, test, config = load_dataset(ds, data_dir)
        tv = np.concatenate([train, val])

        for pl in config['prediction_lengths']:
            print(f"\n{ds} H={pl}")
            t0 = time.time()
            kb_w, kb_f, idx = build_kb(extractor, tv, 512, pl, stride=64)
            print(f"  KB: {len(kb_w)} entries ({time.time()-t0:.0f}s)")

            t0 = time.time()
            base, rcaf, dense = eval_perwindow(
                extractor, test, 512, pl, kb_w, kb_f, idx)
            elapsed = time.time() - t0
            n = len(base)

            if n == 0:
                print(f"  No test windows (test too short for ws=512 + H={pl})")
                continue

            bl_mse = float(base.mean())
            rc_mse = float(rcaf.mean())
            de_mse = float(dense.mean())
            rc_imp = (bl_mse - rc_mse) / bl_mse * 100
            de_imp = (bl_mse - de_mse) / bl_mse * 100

            print(f"  Eval: {n} windows, {elapsed:.0f}s")
            print(f"  baseline:    MSE={bl_mse:.4f}")
            print(f"  dense_bw2:   MSE={de_mse:.4f} ({de_imp:+.2f}%)")
            print(f"  rcaf:        MSE={rc_mse:.4f} ({rc_imp:+.2f}%)")

            # Ablation results
            for method, mse_arr in [('baseline', base), ('dense_bw2', dense), ('rcaf', rcaf)]:
                all_ablation.append({
                    'dataset': ds, 'pred_len': pl, 'method': method,
                    'mse': float(mse_arr.mean()), 'mse_std': float(mse_arr.std()),
                    'n_windows': n,
                })

            # Paired t-test (RCAF vs baseline)
            if n >= 2:
                t_stat, p_val = stats.ttest_rel(base, rcaf)
                low_power = n < 10
                all_ttests.append({
                    'dataset': ds, 'pred_len': pl, 'n': n,
                    'base_mse': bl_mse, 'rcaf_mse': rc_mse,
                    'improvement': rc_imp, 't_stat': float(t_stat),
                    'p_value': float(p_val), 'low_power': low_power,
                })

    # Save ablation results
    abl_df = pd.DataFrame(all_ablation)
    abl_df.to_csv(os.path.join(results_dir, 'ablation_final.csv'), index=False)
    print(f"\nSaved {results_dir}/ablation_final.csv ({len(abl_df)} rows)")

    # Save t-test results
    ttest_lines = [
        "PAIRED T-TESTS: RCAF vs Chronos-Bolt Baseline",
        "=" * 85,
        f"{'Dataset':<10} {'H':>5} {'N':>5} {'Base MSE':>10} {'RCAF MSE':>10} "
        f"{'Δ%':>7} {'t-stat':>8} {'p-value':>10} {'Sig':>6}",
        "-" * 85,
    ]
    prev_ds = None
    for t in all_ttests:
        if t['dataset'] != prev_ds and prev_ds is not None:
            ttest_lines.append("")
        prev_ds = t['dataset']
        p = t['p_value']
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'ns'
        lp = ' [LP]' if t['low_power'] else ''
        ttest_lines.append(
            f"{t['dataset']:<10} {t['pred_len']:>5} {t['n']:>5} "
            f"{t['base_mse']:>10.4f} {t['rcaf_mse']:>10.4f} "
            f"{t['improvement']:>+6.1f}% {t['t_stat']:>+8.3f} "
            f"{t['p_value']:>10.6f} {sig:>4}{lp}"
        )
    ttest_lines.extend([
        "",
        "Significance: *** p<0.01, ** p<0.05, * p<0.1, ns = not significant",
        "[LP] = Low Power (N < 10 windows)",
    ])
    with open(os.path.join(results_dir, 'paired_ttests.txt'), 'w') as f:
        f.write('\n'.join(ttest_lines) + '\n')
    print(f"Saved {results_dir}/paired_ttests.txt ({len(all_ttests)} tests)")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--device', default='cuda')
    p.add_argument('--datasets', nargs='+', default=None)
    a = p.parse_args()
    run_all(a.data_dir, a.results_dir, a.device, a.datasets)
