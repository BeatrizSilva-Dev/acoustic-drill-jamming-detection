from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

SCRIPT_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_PATH
while not (PROJECT_ROOT / 'data').is_dir():
    parent = PROJECT_ROOT.parent
    if parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not find the project root containing the 'data' folder.")
    PROJECT_ROOT = parent
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
PLOT_DIR = PROJECT_ROOT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    'Common': {'prefix': 'common', 'color': '#2878B5'},
    'Ultrasonic': {'prefix': 'ultrasonic', 'color': '#2A9D8F'},
    'Hybrid': {'prefix': 'hybrid', 'color': '#D66B16'},
}
MODELS = {
    'MLP-AE': {'file_key': 'mlp', 'linestyle': ':', 'linewidth': 1.8},
    'LSTM-AE': {'file_key': 'lstm', 'linestyle': '--', 'linewidth': 1.8},
    'XGBoost': {'file_key': 'xgboost', 'linestyle': '-', 'linewidth': 2.0},
}

SHOW_FIGURE = True

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'savefig.facecolor': 'white',
})


def load_roc_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing ROC file: {path}')
    df = pd.read_csv(path)
    missing = {'fpr', 'tpr'} - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    df = df.copy()
    for col in ('fpr', 'tpr'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['fpr', 'tpr']].isna().any().any() or not np.isfinite(df[['fpr', 'tpr']].to_numpy()).all():
        raise ValueError(f'{path.name}: missing or non-finite ROC coordinates.')
    if len(df) < 2:
        raise ValueError(f'{path.name}: at least two ROC points are required.')
    if not df['fpr'].between(0, 1).all() or not df['tpr'].between(0, 1).all():
        raise ValueError(f'{path.name}: ROC coordinates outside [0, 1].')
    # Stable sorting preserves the original order of vertical ROC segments (equal FPR).
    df = df.sort_values('fpr', kind='stable')
    fpr = df['fpr'].to_numpy(dtype=float)
    tpr = df['tpr'].to_numpy(dtype=float)
    if fpr[0] > 1e-8 or fpr[-1] < (1.0 - 1e-8):
        raise ValueError(f'{path.name}: ROC does not span FPR=0 to FPR=1; check the exported file.')
    return fpr, tpr, float(auc(fpr, tpr))


def plot_all_9_curves_roc():
    curves = []
    for config_name, config in CONFIGS.items():
        for model_name, model in MODELS.items():
            filename = f"{config['prefix']}_{model['file_key']}_roc_data.csv"
            fpr, tpr, area = load_roc_csv(RESULTS_DIR / filename)
            curves.append((config_name, model_name, fpr, tpr, area, filename))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.plot([0, 1], [0, 1], color='#777777', linestyle=(0, (2, 2)),
            linewidth=1.0, label='Chance (0.500)', zorder=1)

    rows = []
    for config_name, model_name, fpr, tpr, area, filename in curves:
        config = CONFIGS[config_name]
        model = MODELS[model_name]
        ax.plot(fpr, tpr, color=config['color'], linestyle=model['linestyle'],
                linewidth=model['linewidth'], alpha=0.95, zorder=3,
                label=f'{config_name[:4]}. {model_name} ({area:.3f})')
        rows.append({'configuration': config_name, 'model': model_name,
                     'auc': area, 'roc_file': filename})

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False Positive Rate',
           ylabel='True Positive Rate')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, linestyle=':', linewidth=0.55, alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.99, 0.02),
        ncol=1,
        fontsize=6.8,
        handlelength=2.0,
        labelspacing=0.22,
        borderaxespad=0.2,
        frameon=True,
        facecolor='white',
        edgecolor='none',
        framealpha=0.88
    )

    pdf_path = PLOT_DIR / 'roc_9_curves_column_width.pdf'
    png_path = PLOT_DIR / 'roc_9_curves_column_width.png'
    auc_path = PLOT_DIR / 'roc_auc_values_used_for_plot.csv'

    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    pd.DataFrame(rows).to_csv(auc_path, index=False)

    print('\nAUC recomputed from the nine saved ROC files:')
    print(pd.DataFrame(rows)[['configuration', 'model', 'auc']].to_string(
        index=False, formatters={'auc': '{:.4f}'.format}))
    print(f'\nPDF: {pdf_path}\nPNG: {png_path}\nAUC CSV: {auc_path}')

    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    plot_all_9_curves_roc()