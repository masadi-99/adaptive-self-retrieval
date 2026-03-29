"""Generate publication-quality figures and tables."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'font.family': 'serif',
})

RESULTS_DIR = '../results'
FIGURES_DIR = '../figures'


def plot_improvement_bars():
    """Main result: RCAF MSE improvement per dataset per horizon."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))
    df['method'] = df['method'].str.strip()

    datasets = ['ETTh1', 'ETTm1', 'ETTm2', 'Weather', 'ETTh2', 'ECL', 'Traffic']
    horizons = [96, 192, 336, 720]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    # Use 'rcaf' as primary method, fall back to 'dense_ens_bw2' if not available
    primary = 'rcaf' if 'rcaf' in df.method.values else 'dense_ens_bw2'

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.18

    for i, h in enumerate(horizons):
        vals = []
        for ds in datasets:
            bl = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='baseline')]['mse']
            en = df[(df.dataset==ds)&(df.pred_len==h)&(df.method==primary)]['mse']
            if len(bl)>0 and len(en)>0:
                vals.append((bl.values[0]-en.values[0])/bl.values[0]*100)
            else:
                vals.append(0)
        ax.bar(x + i*width, vals, width, label=f'H={h}', color=colors[i], alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('MSE Improvement (%)')
    ax.set_title(f'RCAF: MSE Improvement over Chronos-Bolt')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(title='Horizon', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'main_results.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'main_results.png'), dpi=150, bbox_inches='tight')
    print("Saved main_results")
    plt.close()


def plot_ablation():
    """Ablation: RCAF vs CRC-only vs CGPF-only vs dense ensemble."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))
    df['method'] = df['method'].str.strip()

    methods = ['dense_bw2', 'rcaf_crc_only', 'rcaf_cgpf_only', 'rcaf']
    nice = {'dense_bw2': 'Scalar Ensemble', 'rcaf_crc_only': 'CRC Only',
            'rcaf_cgpf_only': 'CGPF Only', 'rcaf': 'RCAF (Full)'}
    colors_m = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']

    available = [m for m in methods if m in df.method.values]
    if len(available) < 2:
        print("Not enough methods for ablation plot")
        return

    datasets = ['ETTh1', 'ETTm1', 'ETTm2', 'Weather', 'ETTh2', 'ECL', 'Traffic']
    datasets = [d for d in datasets if d in df.dataset.values]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.18

    for i, m in enumerate(available):
        vals = []
        for ds in datasets:
            bl = df[(df.dataset==ds)&(df.pred_len==96)&(df.method=='baseline')]['mse']
            mv = df[(df.dataset==ds)&(df.pred_len==96)&(df.method==m)]['mse']
            if len(bl)>0 and len(mv)>0:
                vals.append((bl.values[0]-mv.values[0])/bl.values[0]*100)
            else:
                vals.append(0)
        ax.bar(x + i*width, vals, width, label=nice.get(m,m),
               color=colors_m[i % len(colors_m)], alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('MSE Improvement (%)')
    ax.set_title('Ablation: RCAF Components (H=96)')
    ax.set_xticks(x + width * (len(available)-1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation.png'), dpi=150, bbox_inches='tight')
    print("Saved ablation")
    plt.close()


def plot_streaming():
    """Streaming learning curves."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    datasets = ['ETTh1', 'ETTm2', 'Weather']
    available = [d for d in datasets
                 if os.path.exists(os.path.join(RESULTS_DIR, f'streaming_final_{d}.npy'))]
    if not available:
        print("No streaming results")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
    if len(available) == 1: axes = [axes]

    for ax, ds in zip(axes, available):
        data = np.load(os.path.join(RESULTS_DIR, f'streaming_final_{ds}.npy'),
                       allow_pickle=True).item()
        n = len(data['cumulative_mse'])
        cum_asr = data['cumulative_mse']
        cum_bl = np.cumsum(data['baseline_mses']) / np.arange(1, n+1)

        ax.plot(range(n), cum_asr, label='RCAF (online)', color='#2196F3', linewidth=2)
        ax.plot(range(n), cum_bl, label='Chronos-Bolt', color='#F44336',
                linewidth=2, linestyle='--')
        ax.set_xlabel('Windows Processed')
        ax.set_ylabel('Cumulative MSE')
        ax.set_title(ds)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Online Streaming: Cumulative MSE', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'streaming.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'streaming.png'), dpi=150, bbox_inches='tight')
    print("Saved streaming")
    plt.close()


def print_results_table():
    """Print comprehensive results table."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))
    df['method'] = df['method'].str.strip()

    primary = 'rcaf' if 'rcaf' in df.method.values else 'dense_ens_bw2'

    print(f'\nRCAF Results (MSE % improvement over Chronos-Bolt baseline)')
    print(f'{"Dataset":<10} {"H=96":>8} {"H=192":>8} {"H=336":>8} {"H=720":>8} {"Avg":>8}')
    print('-'*55)
    for ds in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','ECL','Traffic']:
        vals = []
        for h in [96,192,336,720]:
            bl = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='baseline')]['mse']
            mv = df[(df.dataset==ds)&(df.pred_len==h)&(df.method==primary)]['mse']
            if len(bl)>0 and len(mv)>0:
                vals.append((bl.values[0]-mv.values[0])/bl.values[0]*100)
            else:
                vals.append(0)
        print(f'{ds:<10} {vals[0]:+7.1f}% {vals[1]:+7.1f}% {vals[2]:+7.1f}% {vals[3]:+7.1f}% {np.mean(vals):+7.1f}%')


def generate_all():
    print("Generating main results...")
    plot_improvement_bars()
    print("Generating ablation...")
    plot_ablation()
    print("Generating streaming...")
    plot_streaming()
    print("Generating results table...")
    print_results_table()
    print("\nDone!")


if __name__ == '__main__':
    generate_all()
