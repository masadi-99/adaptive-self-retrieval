"""Generate publication-quality figures and tables."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif',
})

RESULTS_DIR = '../results'
FIGURES_DIR = '../figures'


def plot_improvement_bars():
    """Main result: MSE improvement per dataset per horizon."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))

    datasets = ['ETTh1', 'ETTm1', 'ETTm2', 'Weather', 'ETTh2', 'ECL', 'Traffic']
    horizons = [96, 192, 336, 720]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.18

    for i, h in enumerate(horizons):
        vals = []
        for ds in datasets:
            bl = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='baseline')]['mse'].values
            en = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='ensemble_bw2')]['mse'].values
            if len(bl)>0 and len(en)>0:
                vals.append((bl[0]-en[0])/bl[0]*100)
            else:
                vals.append(0)
        ax.bar(x + i*width, vals, width, label=f'H={h}', color=colors[i], alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=3.5+width, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('MSE Improvement (%)')
    ax.set_title('Adaptive Self-Retrieval: MSE Improvement over Chronos-Bolt')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(title='Horizon', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(1.5, max(ax.get_ylim())*0.9, 'Improves', fontsize=10, ha='center', color='green', alpha=0.7)
    ax.text(5.5, min(ax.get_ylim())*0.5, 'Degrades', fontsize=10, ha='center', color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'main_results.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'main_results.png'), dpi=150, bbox_inches='tight')
    print("Saved main_results")
    plt.close()


def plot_bw_sensitivity():
    """Base weight sensitivity analysis."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))

    datasets = ['ETTh1', 'ETTm1', 'ETTm2', 'Weather']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, ds in zip(axes, datasets):
        for h in [96, 192, 336]:
            bws = [2, 3, 5]
            imps = []
            bl = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='baseline')]['mse'].values[0]
            for bw in bws:
                en = df[(df.dataset==ds)&(df.pred_len==h)&(df.method==f'ensemble_bw{bw}')]['mse'].values
                if len(en) > 0:
                    imps.append((bl-en[0])/bl*100)
                else:
                    imps.append(0)
            ax.plot(bws, imps, 'o-', label=f'H={h}')
        ax.set_xlabel('Base Weight')
        ax.set_ylabel('MSE Improvement (%)')
        ax.set_title(ds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    plt.suptitle('Sensitivity to Base Weight', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'bw_sensitivity.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'bw_sensitivity.png'), dpi=150, bbox_inches='tight')
    print("Saved bw_sensitivity")
    plt.close()


def plot_streaming():
    """Streaming learning curves."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    datasets = ['ETTh1', 'ETTm2', 'Weather']
    available = [d for d in datasets
                 if os.path.exists(os.path.join(RESULTS_DIR, f'streaming_final_{d}.npy'))]
    if not available:
        print("No streaming results found")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, ds in zip(axes, available):
        data = np.load(os.path.join(RESULTS_DIR, f'streaming_final_{ds}.npy'),
                       allow_pickle=True).item()
        n = len(data['cumulative_mse'])

        # Smooth with rolling window
        window = min(20, n//5)
        if window > 1:
            from scipy.ndimage import uniform_filter1d
            smooth_asr = uniform_filter1d(np.array(data['window_mses']), window)
            smooth_bl = uniform_filter1d(np.array(data['baseline_mses']), window)
            # Cumulative of smoothed
            cum_asr = np.cumsum(smooth_asr) / np.arange(1, n+1)
            cum_bl = np.cumsum(smooth_bl) / np.arange(1, n+1)
        else:
            cum_asr = data['cumulative_mse']
            cum_bl = np.cumsum(data['baseline_mses']) / np.arange(1, n+1)

        ax.plot(range(n), cum_asr, label='ASR (online)', color='#2196F3', linewidth=2)
        ax.plot(range(n), cum_bl, label='Chronos-Bolt', color='#F44336', linewidth=2, linestyle='--')
        ax.set_xlabel('Windows Processed')
        ax.set_ylabel('Cumulative MSE')
        ax.set_title(ds)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Online Streaming: Cumulative MSE Over Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'streaming.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'streaming.png'), dpi=150, bbox_inches='tight')
    print("Saved streaming")
    plt.close()


def print_latex_table():
    """LaTeX table for paper."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_final.csv'))

    print("\n\\begin{table}[t]")
    print("\\centering")
    print("\\caption{MSE comparison: Chronos-Bolt baseline vs. ASR (ensemble, bw=2). "
          "\\textbf{Bold} = best.}")
    print("\\begin{tabular}{l|cccc}")
    print("\\toprule")
    print("Dataset & H=96 & H=192 & H=336 & H=720 \\\\")
    print("\\midrule")

    for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
        row_bl = f"{ds} (Baseline)"
        row_en = f"{ds} (ASR)"
        for h in [96, 192, 336, 720]:
            bl = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='baseline')]['mse'].values[0]
            en = df[(df.dataset==ds)&(df.pred_len==h)&(df.method=='ensemble_bw2')]['mse'].values[0]
            if bl <= en:
                row_bl += f" & \\textbf{{{bl:.2f}}}"
                row_en += f" & {en:.2f}"
            else:
                row_bl += f" & {bl:.2f}"
                row_en += f" & \\textbf{{{en:.2f}}}"
        print(row_bl + " \\\\")
        print(row_en + " \\\\")
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_all():
    print("Generating main results bar chart...")
    plot_improvement_bars()
    print("Generating base weight sensitivity...")
    plot_bw_sensitivity()
    print("Generating streaming curves...")
    plot_streaming()
    print("Generating LaTeX table...")
    print_latex_table()
    print("\nDone!")


if __name__ == '__main__':
    generate_all()
