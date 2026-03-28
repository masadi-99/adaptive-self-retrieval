"""
Run baseline methods for fair comparison.
Vanilla Chronos-Bolt without any retrieval.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline
from datasets_loader import load_dataset, DATASET_CONFIGS


def run_chronos_baseline(
    test: np.ndarray,
    context_length: int,
    prediction_length: int,
    pipeline: ChronosBoltPipeline,
    stride: int = None,
) -> dict:
    """Vanilla Chronos-Bolt without any retrieval."""
    if stride is None:
        stride = prediction_length

    mse_list, mae_list = [], []
    positions = list(range(0, len(test) - context_length - prediction_length + 1, stride))

    for i, start in enumerate(positions):
        context = torch.tensor(
            test[start:start + context_length], dtype=torch.float32
        ).unsqueeze(0)
        gt = test[start + context_length:start + context_length + prediction_length]

        quantiles, mean = pipeline.predict_quantiles(
            context, prediction_length, quantile_levels=[0.5]
        )
        pred = mean[0].numpy()

        mse_list.append(float(np.mean((pred - gt) ** 2)))
        mae_list.append(float(np.mean(np.abs(pred - gt))))

        if (i + 1) % 10 == 0 or (i + 1) == len(positions):
            print(f"    Baseline window {i+1}/{len(positions)}")

    return {
        'mse': float(np.mean(mse_list)),
        'mae': float(np.mean(mae_list)),
        'mse_std': float(np.std(mse_list)),
        'mae_std': float(np.std(mae_list)),
        'n_windows': len(mse_list),
    }


def run_all_baselines(
    data_dir: str = './data',
    results_dir: str = './results',
    model_name: str = "amazon/chronos-bolt-small",
    device: str = "cpu",
    datasets: list = None,
):
    """Run vanilla Chronos baseline on all datasets."""
    os.makedirs(results_dir, exist_ok=True)

    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name, device_map=device, dtype=torch.float32
    )

    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    context_length = 512
    results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Baseline: {dataset_name}")
        print(f"{'='*60}")

        train, val, test, config = load_dataset(dataset_name, data_dir)

        for pred_len in config['prediction_lengths']:
            t0 = time.time()
            metrics = run_chronos_baseline(
                test, context_length, pred_len, pipeline
            )
            elapsed = time.time() - t0

            results.append({
                'dataset': dataset_name,
                'pred_len': pred_len,
                'method': 'chronos_bolt_vanilla',
                **metrics,
                'time_sec': elapsed,
            })

            print(f"  H={pred_len}: MSE={metrics['mse']:.4f}, "
                  f"MAE={metrics['mae']:.4f}, Time={elapsed:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'baseline_results.csv'), index=False)
    print(f"\nBaseline results saved to {results_dir}/baseline_results.csv")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--model', default='amazon/chronos-bolt-base')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()

    run_all_baselines(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        model_name=args.model,
        device=args.device,
        datasets=args.datasets,
    )
