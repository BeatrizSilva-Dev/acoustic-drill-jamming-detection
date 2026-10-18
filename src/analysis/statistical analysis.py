import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = SCRIPT_PATH

while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    parent = os.path.dirname(PROJECT_ROOT)
    if parent == PROJECT_ROOT:
        raise FileNotFoundError(
            "Não foi possível localizar o diretório raiz contendo a pasta 'data'."
        )
    PROJECT_ROOT = parent


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# ARQUIVOS DAS TRÊS CONFIGURAÇÕES
# ============================================================

FILES = {
    "Common": {
        "MLP-AE": "results_autoencoder_regular_option_A.csv",
        "LSTM-AE": "results_LSTM_regular_option_A.csv",
        "XGBoost": "results_xgboost_regular_option_A.csv",
    },
    "Ultrasonic": {
        "MLP-AE": "results_autoencoder_ultrasonic_option_A.csv",
        "LSTM-AE": "results_LSTM_ultrasonic_option_A.csv",
        "XGBoost": "results_xgboost_ultrasonic_option_A.csv",
    },
    "Hybrid": {
        "MLP-AE": "results_autoencoder_hybrid_option_A.csv",
        "LSTM-AE": "results_LSTM_hybrid_option_A.csv",
        "XGBoost": "results_xgboost_hybrid_option_A.csv",
    }
}


# ============================================================
# ESTILO IEEE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ============================================================
# F1-SCORE POR BROCA
# ============================================================

def extract_true_f1_per_drill(df, model):
    f1_results = []
    label_col = "label" if "label" in df.columns else "label_real"
    pred_col = "prediction"

    for drill in sorted(df["drill"].unique()):
        df_drill = df[df["drill"] == drill].copy()

        # Remove a região intermediária (-1) do protocolo 50/30/20
        df_drill = df_drill[df_drill[label_col].isin([0, 1])]

        if df_drill.empty:
            continue

        y_true = df_drill[label_col].astype(int).values
        y_pred = df_drill[pred_col].astype(int).values

        score = f1_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        f1_results.append({
            "drill": drill,
            "f1_score": score,
            "model": model
        })

    return pd.DataFrame(f1_results)


# ============================================================
# WILCOXON + TAMANHO DE EFEITO (r)
# ============================================================

def wilcoxon_f1_comparison(df_model_a, df_model_b):
    paired = pd.merge(
        df_model_a[["drill", "f1_score"]],
        df_model_b[["drill", "f1_score"]],
        on="drill",
        suffixes=("_a", "_b")
    )

    x = paired["f1_score_a"].astype(float).values
    y = paired["f1_score_b"].astype(float).values

    differences = x - y
    nonzero = differences != 0

    x_effective = x[nonzero]
    y_effective = y[nonzero]

    n_pairs = len(paired)
    n_effective = len(x_effective)

    if n_effective == 0:
        return {
            "p_value": 1.0,
            "effect_size_r": 0.0,
            "n_pairs": n_pairs,
            "n_effective": 0
        }

    result = wilcoxon(
        x_effective,
        y_effective,
        zero_method="wilcox",
        method="approx"
    )

    z = result.zstatistic
    r = abs(z) / np.sqrt(n_effective)

    return {
        "p_value": result.pvalue,
        "effect_size_r": r,
        "n_pairs": n_pairs,
        "n_effective": n_effective
    }


# ============================================================
# CORREÇÃO DE HOLM-BONFERRONI
# ============================================================

def holm_correction(p_values):
    """
    Calcula a correção passo a passo de Holm-Bonferroni,
    assegurando a monotonicidade dos p-valores ajustados.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.empty(m)
    for i, p in enumerate(sorted_p):
        adjusted[i] = p * (m - i)

    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(m)
    result[order] = adjusted
    return result


# ============================================================
# CARREGAR OS TRÊS MODELOS DE UMA CONFIGURAÇÃO
# ============================================================

def load_configuration_f1(configuration):
    files = FILES[configuration]

    df_mlp_raw = pd.read_csv(os.path.join(RESULTS_DIR, files["MLP-AE"]))
    df_lstm_raw = pd.read_csv(os.path.join(RESULTS_DIR, files["LSTM-AE"]))
    df_xgb_raw = pd.read_csv(os.path.join(RESULTS_DIR, files["XGBoost"]))

    df_f1_mlp = extract_true_f1_per_drill(df_mlp_raw, "MLP-AE")
    df_f1_lstm = extract_true_f1_per_drill(df_lstm_raw, "LSTM-AE")
    df_f1_xgb = extract_true_f1_per_drill(df_xgb_raw, "XGBoost")

    return df_f1_mlp, df_f1_lstm, df_f1_xgb


# ============================================================
# GERAR BOXPLOT
# ============================================================

def generate_boxplot(configuration, df_f1_mlp, df_f1_lstm, df_f1_xgb):
    df_plot = pd.concat([df_f1_lstm, df_f1_mlp, df_f1_xgb], axis=0).reset_index(drop=True)

    palette = {
        "LSTM-AE": "#7209b7",
        "MLP-AE": "#2ec4b6",
        "XGBoost": "#e67e22"
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    sns.boxplot(
        x="model", y="f1_score", data=df_plot,
        order=["LSTM-AE", "MLP-AE", "XGBoost"],
        palette=palette, width=0.45, linewidth=1.2, fliersize=0, ax=ax
    )

    sns.stripplot(
        x="model", y="f1_score", data=df_plot,
        order=["LSTM-AE", "MLP-AE", "XGBoost"],
        color="black", alpha=0.4, jitter=0.15, size=4, ax=ax
    )

    ax.set_ylabel("F1-score per Drill Unit", fontweight="bold")
    ax.set_xlabel("Detection Architecture", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(configuration, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine()
    plt.tight_layout()

    filename_base = f"{configuration.lower()}_boxplot_f1_per_drill_IEEE"
    plt.savefig(os.path.join(PLOT_DIR, filename_base + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.savefig(os.path.join(PLOT_DIR, filename_base + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()

    print(f"[BOXPLOT] {configuration} salvo em {PLOT_DIR}.")


# ============================================================
# TABELA ESTATÍSTICA: CONFIGURATION, P-VALUE, EFFECT SIZE, HOLM
# ============================================================

def calculate_all_statistics():
    configurations = ["Common", "Ultrasonic", "Hybrid"]
    raw_results = []
    raw_p_values = []

    # Wilcoxon pareado: LSTM-AE vs XGBoost
    for config in configurations:
        df_f1_mlp, df_f1_lstm, df_f1_xgb = load_configuration_f1(config)
        stats = wilcoxon_f1_comparison(df_f1_lstm, df_f1_xgb)

        raw_p_values.append(stats["p_value"])
        raw_results.append({
            "Configuration": config,
            "p_value": stats["p_value"],
            "effect_size_r": stats["effect_size_r"]
        })

    # Ajuste de Holm-Bonferroni
    adjusted_p_values = holm_correction(raw_p_values)

    for i in range(len(raw_results)):
        raw_results[i]["holm_adjusted_p"] = adjusted_p_values[i]

    df_stats = pd.DataFrame(raw_results)

    # ========================================================
    # EXIBIÇÃO FORMATADA IDÊNTICA À TABELA DO ARTIGO
    # ========================================================
    print("\n" + "=" * 70)
    print("PAIRED WILCOXON & HOLM-ADJUSTED P-VALUES (LSTM-AE vs XGBoost)")
    print("=" * 70)

    display_df = pd.DataFrame({
        "Configuration": df_stats["Configuration"],
        "p-value": df_stats["p_value"].map(lambda x: f"{x:.5f}" if x < 0.01 else f"{x:.4f}"),
        "Effect Size (r)": df_stats["effect_size_r"].map(lambda x: f"{x:.3f}"),
        "Holm-adjusted p": df_stats["holm_adjusted_p"].map(lambda x: f"{x:.5f}" if x < 0.01 else f"{x:.4f}")
    })
    print(display_df.to_string(index=False))

    # Gravação do CSV
    csv_path = os.path.join(RESULTS_DIR, "table_statistical_holm_comparison.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"\n[STATISTICS] CSV salvo em:\n {csv_path}")

    # ========================================================
    # EXPORTAÇÃO EM LATEX (PADRÃO OVERLEAF)
    # ========================================================
    latex_path = os.path.join(RESULTS_DIR, "table_statistical_holm_comparison.tex")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired Wilcoxon Signed-Rank Test and Holm-Adjusted $p$-values (LSTM-AE vs. XGBoost).}",
        r"\label{tab:holm_statistical_comparison}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Configuration} & \textbf{$p$-value} & \textbf{Effect Size ($r$)} & \textbf{Holm-adjusted $p$} \\",
        r"\hline"
    ]

    for _, row in df_stats.iterrows():
        p_val_fmt = f"{row['p_value']:.5f}" if row['p_value'] < 0.01 else f"{row['p_value']:.4f}"
        r_fmt = f"{row['effect_size_r']:.3f}"
        h_p_fmt = f"{row['holm_adjusted_p']:.5f}" if row['holm_adjusted_p'] < 0.01 else f"{row['holm_adjusted_p']:.4f}"

        latex_lines.append(f"{row['Configuration']} & {p_val_fmt} & {r_fmt} & {h_p_fmt} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"[STATISTICS] LaTeX salvo em:\n {latex_path}\n")
    return df_stats


# ============================================================
# MAIN
# ============================================================

def generate_all_results():
    configurations = ["Common", "Ultrasonic", "Hybrid"]

    print("\n" + "=" * 80)
    print("ANÁLISE DE F1-SCORE POR BROCA E BOXPLOTS")
    print("=" * 80)

    for configuration in configurations:
        df_f1_mlp, df_f1_lstm, df_f1_xgb = load_configuration_f1(configuration)
        generate_boxplot(configuration, df_f1_mlp, df_f1_lstm, df_f1_xgb)

    df_stats = calculate_all_statistics()
    return df_stats


if __name__ == "__main__":
    try:
        generate_all_results()
        print("=" * 80)
        print("ANÁLISE ESTATÍSTICA CONCLUÍDA COM SUCESSO.")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        raise