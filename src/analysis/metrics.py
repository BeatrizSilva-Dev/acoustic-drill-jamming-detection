import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# ESTILO VISUAL DO ARTIGO (IEEE / SAFEPROCESS)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# RESOLUÇÃO DINÂMICA DE DIRETÓRIOS
# ============================================================
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH
while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    parent = os.path.dirname(PROJECT_ROOT)
    if parent == PROJECT_ROOT:
        raise FileNotFoundError("Diretório raiz contendo a pasta 'data' não foi localizado.")
    PROJECT_ROOT = parent

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# MAPEAMENTO DOS ARQUIVOS
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
    },
}

def load_binary_data(path):
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else "label_real"
    # Protocolo 50/30/20: avalia 0 (normal) e 1 (falha), excluindo -1 (transição)
    df = df[df[label_col].isin([0, 1])].copy()
    df[label_col] = df[label_col].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    return df, label_col

# ============================================================
# MÉTRICAS AMOSTRA A AMOSTRA (TABELA 2)
# ============================================================
def calculate_metrics(df, label_col, model):
    y_true = df[label_col].values
    y_pred = df["prediction"].values

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Correção da coluna do erro contínuo
    score_col = "score" if model == "XGBoost" else "mse"

    if score_col in df.columns:
        df_auc = df.dropna(subset=[score_col])
        if df_auc[label_col].nunique() == 2:
            auc_value = roc_auc_score(
                df_auc[label_col].astype(int),
                df_auc[score_col].astype(float),
            )
        else:
            auc_value = np.nan
    else:
        auc_value = np.nan

    return f1, prec, recall, accuracy, auc_value

# ============================================================
# LEAD-TIME ANTES DA REGIÃO DE 80% (SEM EXCLUIR FALHAS)
# ============================================================
def calculate_mlt(df_full, persistence_window=3):
    """Calcula a antecedência em furos usando a série temporal contínua da broca."""
    lead_times = []

    for drill in sorted(df_full["drill"].unique()):
        sub_drill = df_full[df_full["drill"] == drill].sort_values("hole").reset_index(drop=True)
        n_holes = len(sub_drill)
        idx_80 = int(np.floor(0.80 * n_holes))

        # Considera apenas alarmes persistentes ocorridos antes da marca dos 80%
        preds_pre80 = sub_drill["prediction"].astype(int).values[:idx_80]

        first_alert_idx = None
        for i in range(len(preds_pre80) - persistence_window + 1):
            if np.all(preds_pre80[i : i + persistence_window] == 1):
                first_alert_idx = i
                break

        if first_alert_idx is not None:
            drill_lead_time = idx_80 - first_alert_idx
        else:
            drill_lead_time = 0

        lead_times.append(drill_lead_time)

    return np.mean(lead_times)

# ============================================================
# MATRIZ REGIONAL POR BROCA (CRITÉRIO 50% / 20% DO ARTIGO ANTERIOR)
# ============================================================
def compute_regional_cm(df):
    regional_results = []
    for drill in df["drill"].unique():
        sub_drill = df[df["drill"] == drill].sort_values("hole").reset_index(drop=True)
        n_holes = len(sub_drill)

        idx_50 = int(np.floor(0.50 * n_holes))
        idx_80 = int(np.floor(0.80 * n_holes))

        # Região normal: falso alarme se disparou nos primeiros 50%
        pred_normal = int(np.any(sub_drill["prediction"].iloc[:idx_50] == 1))
        regional_results.append({"y_true": 0, "y_pred": pred_normal})

        # Região crítica: alarme correto se disparou nos últimos 20%
        pred_anomaly = int(np.any(sub_drill["prediction"].iloc[idx_80:] == 1))
        regional_results.append({"y_true": 1, "y_pred": pred_anomaly})

    df_regional = pd.DataFrame(regional_results)
    return confusion_matrix(df_regional["y_true"], df_regional["y_pred"], labels=[0, 1])

# ============================================================
# GERAÇÃO DAS FIGURAS DAS MATRIZES
# ============================================================
def plot_regional_matrices(cms, configuration):
    # Figura dos Autoencoders (lado a lado)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

    sns.heatmap(
        cms["MLP-AE"], annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[0],
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=["No Alert", "Alert"], yticklabels=["Normal", "Anomaly"]
    )
    axes[0].set_xlabel("Predicted Label", fontsize=10, weight="bold")
    axes[0].set_ylabel("Ground Truth Label", fontsize=10, weight="bold")
    axes[0].set_title("(a) MLP-AE.", fontsize=11, style="italic", pad=8)

    sns.heatmap(
        cms["LSTM-AE"], annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[1],
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=["No Alert", "Alert"], yticklabels=["Normal", "Anomaly"]
    )
    axes[1].set_xlabel("Predicted Label", fontsize=10, weight="bold")
    axes[1].set_ylabel("True Label", fontsize=10, weight="bold")
    axes[1].set_title("(b) LSTM-AE.", fontsize=11, style="italic", pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"fig_confusion_autoencoders_{configuration.lower()}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOT_DIR, f"fig_confusion_autoencoders_{configuration.lower()}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figura individual do XGBoost
    fig_xgb, ax_xgb = plt.subplots(figsize=(3.6, 3.2))
    sns.heatmap(
        cms["XGBoost"], annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax_xgb,
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=["No Alert", "Alert"], yticklabels=["Normal (50%)", "Anomaly (20%)"]
    )
    ax_xgb.set_xlabel("Predicted Label", fontsize=10, weight="bold")
    ax_xgb.set_ylabel("Ground Truth", fontsize=10, weight="bold")
    ax_xgb.set_title("XGBoost (50/20 criteria)", fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"fig_confusion_xgboost_{configuration.lower()}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOT_DIR, f"fig_confusion_xgboost_{configuration.lower()}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# LOOP PRINCIPAL
# ============================================================
table_results = []

for configuration, models in FILES.items():
    print("\n" + "=" * 80)
    print(f"PROCESSANDO DOMÍNIO: {configuration.upper()}")
    print("=" * 80)

    domain_cms = {}

    for model, filename in models.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"[AVISO] Arquivo não encontrado: {path}")
            continue

        # 1. Carrega o dataset completo com todos os furos (inclusive transição -1)
        df_full = pd.read_csv(path)
        label_col = "label" if "label" in df_full.columns else "label_real"
        df_full[label_col] = df_full[label_col].astype(int)
        df_full["prediction"] = df_full["prediction"].astype(int)

        # 2. Calcula o Lead-Time na série temporal contínua da broca (janela = 3)
        mlt = calculate_mlt(df_full, persistence_window=3)

        # 3. Calcula a matriz de confusão regional por broca (50% normal vs 20% anomalia)
        cm_reg = compute_regional_cm(df_full)
        domain_cms[model] = cm_reg

        # 4. Filtra a zona de transição (-1) apenas para as métricas pontuais de classificação
        df_eval = df_full[df_full[label_col].isin([0, 1])].copy()
        f1, prec, recall, accuracy, auc_value = calculate_metrics(df_eval, label_col, model)

        table_results.append({
            "Configuration": configuration,
            "Model": model,
            "F1-score": f1,
            "Precision": prec,
            "Recall": recall,
            "Accuracy": accuracy,
            "AUC": auc_value,
            "MLT (Holes)": mlt,
        })

    if len(domain_cms) == 3:
        plot_regional_matrices(domain_cms, configuration)
# ============================================================
# CONSOLIDAÇÃO DA TABELA 2
# ============================================================
df_table = pd.DataFrame(table_results)
configuration_order = ["Common", "Ultrasonic", "Hybrid"]
model_order = ["LSTM-AE", "MLP-AE", "XGBoost"]

df_table["Configuration"] = pd.Categorical(df_table["Configuration"], categories=configuration_order, ordered=True)
df_table["Model"] = pd.Categorical(df_table["Model"], categories=model_order, ordered=True)
df_table = df_table.sort_values(["Configuration", "Model"]).reset_index(drop=True)

print("\n" + "=" * 100)
print("TABELA 2 — DESEMPENHO DOS MODELOS")
print("=" * 100)
print(df_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Gravação em CSV e LaTeX
csv_path = os.path.join(RESULTS_DIR, "table_2_model_performance.csv")
df_table.to_csv(csv_path, index=False)

latex_table = df_table.copy()
for col in ["F1-score", "Precision", "Recall", "Accuracy", "AUC", "MLT (Holes)"]:
    latex_table[col] = latex_table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "--")

latex_path = os.path.join(RESULTS_DIR, "table_2_model_performance.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex_table.to_latex(index=False, escape=False, column_format="llcccccc"))

print(f"\nTabelas salvas com sucesso em:\n - {csv_path}\n - {latex_path}")
print(f"Gráficos das matrizes salvos na pasta: {PLOT_DIR}")