"""
Canonical evaluation for RERC (Retrieve Errors, not Examples).
Produces: ablation_rerc.csv, rerc_metrics.csv, rerc_ttests.txt
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from datasets_loader import load_dataset, DATASET_CONFIGS
from embeddings import ChronosEmbeddingExtractor
from rerc import RERC, ResidualMemoryBank

warnings.filterwarnings('ignore')
np.random.seed(42)


def eval_rerc(extractor, test, ws, pl, rerc_model, unc_mean, unc_std):
    """Evaluate RERC on test set. Returns per-window metrics."""
    positions = list(range(0, len(test) - ws - pl + 1, pl))
    if not positions:
        return []

    results = []
    for s in positions:
        ctx = test[s:s+ws]
        gt = test[s+ws:s+ws+pl]
        base, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        base_mse = float(np.mean((base - gt)**2))

        corrected, meta = rerc_model.intervene(base, ctx, unc, unc_mean, unc_std)
        corr_mse = float(np.mean((corrected - gt)**2))

        results.append({
            'base_mse': base_mse,
            'corr_mse': corr_mse,
            'improvement': (base_mse - corr_mse) / base_mse * 100,
            'intervened': meta['intervention_rate'] > 0,
            'intervention_rate': meta['intervention_rate'],
            'mean_score': meta['mean_score'],
            'n_neighbors': meta['n_neighbors'],
        })
    return results


def compute_intervention_metrics(per_window):
    """Compute RERC-specific intervention quality metrics."""
    if not per_window:
        return {}

    df = pd.DataFrame(per_window)
    n_total = len(df)
    intervened = df[df.intervened]
    n_intervened = len(intervened)

    if n_intervened == 0:
        return {
            'mse_improvement': 0.0,
            'intervention_coverage': 0.0,
            'conditional_gain': 0.0,
            'hir': 0.0,
            'mean_excess_loss': 0.0,
            'n_total': n_total,
            'n_intervened': 0,
        }

    # 1. Overall MSE improvement
    mse_imp = float(df.improvement.mean())

    # 2. Intervention coverage
    coverage = n_intervened / n_total

    # 3. Conditional gain (among intervened windows)
    cond_gain = float(intervened.improvement.mean())

    # 4. Harmful Intervention Rate (HIR)
    harmful = intervened[intervened.corr_mse > intervened.base_mse]
    hir = len(harmful) / n_intervened

    # 5. Mean Excess Loss (among harmful interventions)
    if len(harmful) > 0:
        excess = (harmful.corr_mse - harmful.base_mse) / harmful.base_mse * 100
        mel = float(excess.mean())
    else:
        mel = 0.0

    return {
        'mse_improvement': mse_imp,
        'intervention_coverage': coverage,
        'conditional_gain': cond_gain,
        'hir': hir,
        'mean_excess_loss': mel,
        'n_total': n_total,
        'n_intervened': n_intervened,
        'n_harmful': len(harmful),
    }


def run_all(data_dir='./data', results_dir='./results', device='cuda',
            datasets=None):
    os.makedirs(results_dir, exist_ok=True)
    extractor = ChronosEmbeddingExtractor('amazon/chronos-bolt-small', device)
    print(f"Model loaded. dim={extractor.embedding_dim}")

    if datasets is None:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']

    all_ablation = []
    all_metrics = []
    all_ttests = []

    for ds in datasets:
        train, val, test, config = load_dataset(ds, data_dir)
        tv = np.concatenate([train, val])

        for pl in config['prediction_lengths']:
            print(f"\n{ds} H={pl}")

            # Step 1: Calibrate threshold on val (using train-only memory)
            t0 = time.time()
            rerc = RERC(extractor, k=7, beta=0.4, intervention_threshold=0.15)
            threshold = rerc.calibrate_threshold(val, train, 512, pl, target_hir=0.10)
            cal_time = time.time() - t0
            print(f"  Calibration: {cal_time:.0f}s, threshold={threshold}")

            # Step 2: Build full memory from train+val
            t0 = time.time()
            rerc.memory = ResidualMemoryBank(
                extractor.embedding_dim, rerc.forecast_desc_dim, alpha_context=0.7
            )
            rerc.build_memory(tv, 512, pl, stride=64)
            build_time = time.time() - t0

            # Step 3: Collect test uncertainties
            positions = list(range(0, len(test) - 512 - pl + 1, pl))
            all_uncs = [extractor.predict_with_uncertainty(test[s:s+512], pl)[2]
                       for s in positions]
            unc_mean = np.mean(all_uncs) if all_uncs else 0
            unc_std = np.std(all_uncs) + 1e-8 if all_uncs else 1

            # Step 4: Evaluate
            t0 = time.time()
            per_window = eval_rerc(extractor, test, 512, pl, rerc, unc_mean, unc_std)
            eval_time = time.time() - t0

            if not per_window:
                print(f"  No test windows")
                continue

            # Compute metrics
            metrics = compute_intervention_metrics(per_window)
            metrics.update({'dataset': ds, 'pred_len': pl, 'threshold': threshold})

            print(f"  MSE imp: {metrics['mse_improvement']:+.2f}%, "
                  f"Coverage: {metrics['intervention_coverage']:.0%}, "
                  f"Cond. gain: {metrics['conditional_gain']:+.2f}%, "
                  f"HIR: {metrics['hir']:.0%}, "
                  f"({eval_time:.0f}s)")

            all_metrics.append(metrics)

            # Ablation: baseline MSE
            base_mses = [w['base_mse'] for w in per_window]
            corr_mses = [w['corr_mse'] for w in per_window]
            bl_mse = np.mean(base_mses)
            rc_mse = np.mean(corr_mses)
            imp = (bl_mse - rc_mse) / bl_mse * 100

            all_ablation.append({
                'dataset': ds, 'pred_len': pl, 'method': 'baseline',
                'mse': bl_mse, 'n_windows': len(per_window),
            })
            all_ablation.append({
                'dataset': ds, 'pred_len': pl, 'method': 'rerc',
                'mse': rc_mse, 'n_windows': len(per_window),
            })

            # Paired t-test
            n = len(base_mses)
            if n >= 2:
                t_stat, p_val = stats.ttest_rel(base_mses, corr_mses)
                all_ttests.append({
                    'dataset': ds, 'pred_len': pl, 'n': n,
                    'base_mse': bl_mse, 'rerc_mse': rc_mse,
                    'improvement': imp, 't_stat': float(t_stat),
                    'p_value': float(p_val), 'low_power': n < 10,
                })

    # Save results
    pd.DataFrame(all_ablation).to_csv(
        os.path.join(results_dir, 'ablation_rerc.csv'), index=False)
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(results_dir, 'rerc_metrics.csv'), index=False)

    # T-test file
    lines = [
        "PAIRED T-TESTS: RERC vs Chronos-Bolt Baseline",
        "=" * 90,
        f"{'Dataset':<10} {'H':>5} {'N':>5} {'Base MSE':>10} {'RERC MSE':>10} "
        f"{'Δ%':>7} {'t-stat':>8} {'p-value':>10} {'Sig':>6}",
        "-" * 90,
    ]
    prev_ds = None
    for t in all_ttests:
        if t['dataset'] != prev_ds and prev_ds is not None:
            lines.append("")
        prev_ds = t['dataset']
        p = t['p_value']
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'ns'
        lp = ' [LP]' if t['low_power'] else ''
        lines.append(
            f"{t['dataset']:<10} {t['pred_len']:>5} {t['n']:>5} "
            f"{t['base_mse']:>10.4f} {t['rerc_mse']:>10.4f} "
            f"{t['improvement']:>+6.1f}% {t['t_stat']:>+8.3f} "
            f"{t['p_value']:>10.6f} {sig:>4}{lp}"
        )
    lines.extend(["", "*** p<0.01, ** p<0.05, * p<0.1, ns=not significant, [LP]=N<10"])
    with open(os.path.join(results_dir, 'rerc_ttests.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Print intervention metrics summary
    print(f"\n{'='*80}")
    print("RERC INTERVENTION METRICS")
    print(f"{'='*80}")
    print(f"{'DS':<10} {'H':>5} {'MSE Δ%':>8} {'Cover':>7} {'CondGain':>9} {'HIR':>6} {'Thresh':>7}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['dataset']:<10} {m['pred_len']:>5} "
              f"{m['mse_improvement']:>+7.1f}% {m['intervention_coverage']:>6.0%} "
              f"{m['conditional_gain']:>+8.1f}% {m['hir']:>5.0%} {m['threshold']:>7.2f}")

    print(f"\nSaved: ablation_rerc.csv, rerc_metrics.csv, rerc_ttests.txt")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--device', default='cuda')
    p.add_argument('--datasets', nargs='+', default=None)
    a = p.parse_args()
    run_all(a.data_dir, a.results_dir, a.device, a.datasets)
