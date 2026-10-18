from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "LSTM-AE": "results_LSTM_hybrid.csv",
    "MLP-AE": "results_autoencoder_hybrid.csv",
    "XGBoost": "results_xgboost_hybrid.csv",
}
ORDER = list(MODEL_FILES)
COLORS = {"LSTM-AE": "#7209b7", "MLP-AE": "#2b9f97", "XGBoost": "#e67e22"}
EXPECTED_COUNTS = (20, 458, 305, 153)
STRICT_EXPECTED_COUNTS = True
OUTPUT_STEM = "hybrid_per_drill_F1"


def project_root():
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "data" / "results").is_dir():
            return candidate
    raise FileNotFoundError("The data/results folder was not found in the directories above the script.")


def drill_sort_key(s):
    m = re.search(r"(\d+)$", str(s))
    return (int(m.group(1)) if m else 10**9, str(s))


def load_and_filter(path, model):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "label" not in df and "label_real" in df:
        df = df.rename(columns={"label_real": "label"})
    required = {"drill", "hole", "label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model}: There are missing columns {sorted(missing)} in {path.name}")
    df = df.copy()
    df["drill"] = df["drill"].astype(str).str.lower().str.strip()
    for c in ("hole", "label", "prediction"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["hole", "label"]].isna().any().any():
        raise ValueError(f"{model}: hole/label contains missing or invalid values.")
    if df.duplicated(["drill", "hole"]).any():
        raise ValueError(f"{model}: there are duplicate pairs (drill, hole).")
    if not df["label"].isin([-1, 0, 1]).all():
        raise ValueError(f"{model}: unexpected labels; expected -1, 0 or 1.")
    # Physical hole numbers, not row indices. First five holes reserved for calibration.
    eligible = df.loc[(df["hole"] >= 6) & df["label"].isin([0, 1])].copy()
    if eligible.empty:
        raise ValueError(f"{model}: no eligible events after filter.")
    if eligible["prediction"].isna().any() or not eligible["prediction"].isin([0, 1]).all():
        raise ValueError(f"{model}: missing/non-binary prediction on eligible events.")
    eligible[["hole", "label", "prediction"]] = eligible[["hole", "label", "prediction"]].astype(int)
    return eligible.sort_values(["drill", "hole"]).reset_index(drop=True)
def per_drill_f1(df, model):
    records = []
    for drill, g in df.groupby("drill", sort=False):
        yt = g["label"].to_numpy(dtype=int)
        yp = g["prediction"].to_numpy(dtype=int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        records.append(dict(drill=drill, model=model, f1_score=f1,
                            n_events=len(g), n_normal=int((yt == 0).sum()),
                            n_failure=int((yt == 1).sum()), TN=tn, FP=fp, FN=fn, TP=tp))
    return pd.DataFrame(records)

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    idx = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[idx])
    adjusted = np.empty_like(p)
    adjusted[idx] = np.minimum(1.0, adjusted_sorted)
    return adjusted

def paired_tests(wide):
    pairs = [("LSTM-AE", "MLP-AE"), ("LSTM-AE", "XGBoost"), ("MLP-AE", "XGBoost")]
    rows = []
    for a, b in pairs:
        x, y = wide[a].to_numpy(float), wide[b].to_numpy(float)
        diff = x - y
        effective = int((~np.isclose(diff, 0, rtol=0, atol=1e-12)).sum())
        if effective == 0:
            statistic, p = 0.0, 1.0
        else:
            result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
            statistic, p = float(result.statistic), float(result.pvalue)
        rows.append(dict(configuration="Hybrid", comparison=f"{a} vs {b}",
                         n_pairs=len(x), n_nonzero=effective, W_statistic=statistic,
                         p_raw=p, median_paired_difference=float(np.median(diff))))
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"])
    result["significant_holm_0_05"] = result["p_holm"] < 0.05
    return result


def plot_boxplot(long_df, output_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 3.15))  # ~single IFAC column width
    data = [long_df.loc[long_df.model == m, "f1_score"].to_numpy(float) for m in ORDER]
    bp = ax.boxplot(data, positions=np.arange(1, 4), widths=0.47, patch_artist=True,
                    showfliers=False, whis=1.5,
                    medianprops={"color": "#202020", "linewidth": 1.25},
                    boxprops={"linewidth": 0.85}, whiskerprops={"linewidth": 0.85},
                    capprops={"linewidth": 0.85})
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.68)
    rng = np.random.default_rng(2027)
    for i, m in enumerate(ORDER, start=1):
        vals = long_df.loc[long_df.model == m, "f1_score"].to_numpy(float)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=10,
                   facecolor="#1c1c1c", edgecolor="white", linewidth=0.25,
                   alpha=0.72, zorder=4)
    ax.set_xticks([1, 2, 3], ORDER)
    ax.set_ylabel("Event-level F1 per held-out drill")
    ax.set_ylim(-0.035, 1.035)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlim(0.5, 3.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.65, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.15, top=0.98)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(output_dir / f"{OUTPUT_STEM}.{ext}", bbox_inches="tight", pad_inches=0.035, **kwargs)
    plt.close(fig)


def main():
    root = project_root()
    results_dir = root / "data" / "results"
    out_dir = root / "plots" / "hybrid_boxplot"
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = {m: load_and_filter(results_dir / f, m) for m, f in MODEL_FILES.items()}
    reference = dfs[ORDER[0]][["drill", "hole", "label"]].reset_index(drop=True)
    for m in ORDER[1:]:
        other = dfs[m][["drill", "hole", "label"]].reset_index(drop=True)
        if not reference.equals(other):
            raise ValueError(f"{m}: eligible events/labels do not match {ORDER[0]}.")
    counts = (reference.drill.nunique(), len(reference), int((reference.label == 0).sum()),
              int((reference.label == 1).sum()))
    print(f"LODO Hybrid: {counts[0]} drills; {counts[1]} eligible events "
          f"({counts[2]} normal, {counts[3]} failure-aligned).")
    if counts != EXPECTED_COUNTS:
        msg = f"Counts differ from the expected {EXPECTED_COUNTS}: obtained {counts}."
        if STRICT_EXPECTED_COUNTS:
            raise ValueError(msg + "Check the protocol and the CSVs before publishing.")
        print("WARNING:", msg)
    long_df = pd.concat([per_drill_f1(dfs[m], m) for m in ORDER], ignore_index=True)
    wide = long_df.pivot(index="drill", columns="model", values="f1_score").reindex(columns=ORDER)
    if wide.isna().any().any() or len(wide) != counts[0]:
        raise ValueError("F1 unavailable for some drill/model; invalid pairwise comparison.")
    wide = wide.loc[sorted(wide.index, key=drill_sort_key)]
    long_df.to_csv(results_dir / "hybrid_per_drill_F1.csv", index=False)
    wide.to_csv(results_dir / "hybrid_paired_per_drill_F1.csv", index_label="drill")
    stats = paired_tests(wide)
    stats.to_csv(results_dir / "hybrid_wilcoxon_holm.csv", index=False)
    summary = long_df.groupby("model").f1_score.agg(["count", "mean", "median", "std", "min", "max"]).reindex(ORDER)
    summary.to_csv(results_dir / "hybrid_per_drill_F1_summary.csv")
    plot_boxplot(long_df, out_dir)
    print("\nPer-drill F1 summary:\n", summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nPaired two-sided Wilcoxon:\n",
          stats.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nFigures: {out_dir / (OUTPUT_STEM + '.pdf')} and PNG")
    print(f"Tables: {results_dir}")


if __name__ == "__main__":
    main()
