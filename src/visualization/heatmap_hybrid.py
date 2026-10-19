from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

SCRIPT_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_PATH
while not (PROJECT_ROOT / 'data').is_dir():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not locate the project root containing 'data'.")
    PROJECT_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
OUTPUT_DIR = PROJECT_ROOT / 'plots' / 'heatmap_hybrid'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'XGBoost': 'results_xgboost_hybrid.csv',
    'MLP-AE': 'results_autoencoder_hybrid.csv',
    'LSTM-AE': 'results_LSTM_hybrid.csv',
}
NORM_POINTS = 101
COLORS = ['#D9D9D9', '#9DBBD0', '#F4A261', '#B8A9C9', '#D73027', '#FFE08A']
LABELS = [
    'Calibration (holes 1–5)', 'Normal — No Alert', 'Normal — Alert',
    'Transition (retrospective)', 'Failure-aligned — Alert',
    'Failure-aligned — No Alert',
]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), CMAP.N)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def drill_sort_key(name):
    match = re.search(r'(\d+)$', str(name))
    return (0, int(match.group(1))) if match else (1, str(name))


def drill_short_name(name):
    match = re.search(r'drill_4mm_(\d+)', str(name), re.I)
    return match.group(1).zfill(2) if match else str(name)


def load_results(path):
    df = pd.read_csv(path)
    required = {'drill', 'hole', 'label', 'prediction'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: columns are missing {sorted(missing)}')
    df['drill'] = df['drill'].astype(str).str.lower().str.strip()
    for col in ['hole', 'label', 'prediction']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['hole', 'label']].isna().any().any():
        raise ValueError(f'{path.name}: There is a missing or invalid hole/label.')
    if df.duplicated(['drill', 'hole']).any():
        raise ValueError(f'{path.name}: There are duplicate drill/hole pairs.')
    if not df['label'].isin([-1, 0, 1]).all():
        raise ValueError(f'{path.name}: labels outside of {{-1, 0, 1}}.')
    bad = df[(df['hole'] > 5) & (df['label'].isin([0, 1])) &
             (~df['prediction'].isin([0, 1]))]
    if not bad.empty:
        raise ValueError(f'{path.name}: Invalid prediction in the monitored binary region.')
    return df.sort_values(['drill', 'hole']).reset_index(drop=True)

def categorical_state(row):
    hole = int(row['hole'])
    label = int(row['label'])
    prediction = row['prediction']
    if hole <= 5:
        return 0
    if label == -1:
        return 3
    if pd.isna(prediction) or prediction not in (0, 1):
        raise ValueError(f'Missing prediction for monitored hole {hole}.')
    prediction = int(prediction)
    if label == 0:
        return 2 if prediction == 1 else 1
    return 4 if prediction == 1 else 5

def generate_matrix(df, drills):
    rows = []
    for drill in drills:
        sub = df.loc[df['drill'] == drill].sort_values('hole').reset_index(drop=True)
        if sub.empty:
            raise ValueError(f'Drill bit {drill} missing from the CSV.')
        holes = sub['hole'].to_numpy(dtype=float)
        if not np.array_equal(holes, np.arange(1, len(holes) + 1)):
            raise ValueError(f'{drill}: The sequence of holes must be 1, 2, ..., N, with no gaps.')
        if len(holes) < 6:
            raise ValueError(f'{drill}: fewer than six holes.')
        states = sub.apply(categorical_state, axis=1).to_numpy(dtype=int)
        original_percent = np.linspace(0, 100, len(states))
        target_percent = np.linspace(0, 100, NORM_POINTS)
        indices = np.abs(original_percent[:, None] - target_percent[None, :]).argmin(axis=0)
        rows.append(states[indices])
    return np.asarray(rows, dtype=int)

def plot_heatmap(matrix, drills, model, stem):
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    x_edges = np.arange(NORM_POINTS + 1)
    y_edges = np.arange(len(drills) + 1)
    ax.pcolormesh(x_edges, y_edges, matrix, cmap=CMAP, norm=NORM,
                  edgecolors='#FFFFFF', linewidth=0.45, shading='flat', rasterized=False)
    ax.set_ylim(len(drills), 0)
    ax.set_xlim(0, NORM_POINTS)
    tick_values = np.arange(0, 101, 20)
    ax.set_xticks(tick_values / 100 * NORM_POINTS)
    ax.set_xticklabels([f'{v}%' for v in tick_values], fontsize=9)
    ax.set_yticks(np.arange(len(drills)) + 0.5)
    ax.set_yticklabels([drill_short_name(d) for d in drills], rotation=0, fontsize=8)
    ax.tick_params(axis='both', length=0)
    ax.axvline(0.8 * NORM_POINTS, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel('Normalized Drilling Sequence (%)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Drill Number', fontsize=9, fontweight='bold')
    legend = [mpatches.Patch(facecolor=color, edgecolor='#888888', linewidth=0.25, label=label)
              for color, label in zip(COLORS, LABELS)]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=2, fontsize=7.1, frameon=False, columnspacing=0.8,
              handlelength=1.1, handletextpad=0.45, borderaxespad=0)
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def main():
    frames = {model: load_results(RESULTS_DIR / filename) for model, filename in FILES.items()}
    drills = sorted(frames['XGBoost']['drill'].unique(), key=drill_sort_key)
    reference = frames['XGBoost'][['drill', 'hole', 'label']].sort_values(['drill', 'hole']).reset_index(drop=True)
    for model, df in frames.items():
        check = df[['drill', 'hole', 'label']].sort_values(['drill', 'hole']).reset_index(drop=True)
        if not check.equals(reference):
            raise ValueError(f'{model}: The sequence (drill, hole, label) differs from the XGBoost reference.')
        matrix = generate_matrix(df, drills)
        stem = 'heatmap_' + model.lower().replace('-', '_') + '_hybrid'
        plot_heatmap(matrix, drills, model, stem)
        pd.DataFrame(matrix, index=drills, columns=[f'{p}%' for p in range(NORM_POINTS)]).to_csv(OUTPUT_DIR / f'{stem}_matrix.csv', index_label='drill')
        print(f'{model}: {len(drills)} drills; saved figure: {stem}.pdf/.png')
    print(f'Files in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
