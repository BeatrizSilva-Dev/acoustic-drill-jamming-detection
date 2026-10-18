# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     roc_curve,
# )
#
#
# SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = SCRIPT_PATH
# while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
#     parent = os.path.dirname(PROJECT_ROOT)
#     if parent == PROJECT_ROOT:
#         raise FileNotFoundError("Não foi possível localizar o diretório raiz com a pasta 'data'.")
#     PROJECT_ROOT = parent
#
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
#
# # FUNÇÃO: LOCALIZAR O PRIMEIRO ALARME PERSISTENTE
# def find_persistent_alert(predictions, window=2):
#     """
#     Retorna o índice inicial da primeira sequência consecutiva de alarmes
#     com comprimento igual ou superior a 'window'.
#     """
#     predictions = np.asarray(predictions).astype(int)
#     for i in range(len(predictions) - window + 1):
#         if np.all(predictions[i : i + window] == 1):
#             return i
#     return None
#
# # FUNÇÃO DE AVALIAÇÃO INDIVIDUAL DE CADA MODELO
# def evaluate_single_model(filename, model_name, score_col, persistence_window=2):
#     filepath = os.path.join(RESULTS_DIR, filename)
#     if not os.path.exists(filepath):
#         print(f"[AVISO] Arquivo não encontrado: {filepath}")
#         return None
#
#     df = pd.read_csv(filepath)
#     print(f"\nArquivo: {filename}")
#     print("Colunas disponíveis:", df.columns.tolist())
#     # 1. Filtro do protocolo 50/30/20: exclusão estrita da transição (-1)
# # Ordena cronologicamente e exclui os furos de calibração
#     df = df.sort_values(["drill", "hole"]).copy()
#     df_monitor = df[df["hole"] >= 6].copy()
#
#     # Métricas binárias: somente rótulos 0 e 1
#     df_eval = df_monitor[df_monitor["label"].isin([0, 1])].copy()
#
#     if df_eval.empty:
#         raise ValueError(
#             f"Sem amostras válidas no monitoramento para {model_name}."
#         )
#
#     y_true = df_eval["label"].astype(int).to_numpy()
#     y_pred = df_eval["prediction"].astype(int).to_numpy()
#
#     # 2. Métricas de classificação
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     acc = accuracy_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#
#     # 3. Tratamento para ROC/AUC (remove NaNs gerados pelo SEQ_LEN inicial)
#     y_score_raw = df_eval[score_col].astype(float).values
#     valid_mask = np.isfinite(y_score_raw)
#
#     y_true_auc = y_true[valid_mask]
#     y_score_auc = y_score_raw[valid_mask]
#
#     if len(np.unique(y_true_auc)) < 2:
#         auc = np.nan
#         fpr, tpr = np.array([]), np.array([])
#     else:
#         auc = roc_auc_score(y_true_auc, y_score_auc)
#         fpr, tpr, _ = roc_curve(y_true_auc, y_score_auc)
#
#     # 4. Lead-time corrigido: apenas alarmes persistentes antes dos 80% da vida útil
#     # MLT: monitoramento do furo 6 até o limite de 80% da sequência
#     lead_times = []
#
#     for drill in df["drill"].unique():
#         sub_drill = (
#             df[df["drill"] == drill]
#             .sort_values("hole")
#             .reset_index(drop=True)
#         )
#
#         n_holes = len(sub_drill)
#
#         # Índice final da região anterior/igual ao limite de 80%
#         idx_80 = int(np.floor(0.80 * n_holes))
#
#         # Considera somente furos de monitoramento até o limite de 80%
#         sub_monitor = sub_drill[
#             (sub_drill["hole"] >= 6) &
#             (sub_drill["hole"] <= idx_80)
#             ]
#
#         preds_pre80 = sub_monitor["prediction"].astype(int).to_numpy()
#
#         first_alert_idx = find_persistent_alert(
#             preds_pre80,
#             window=persistence_window
#         )
#
#         if first_alert_idx is not None:
#             # Distância, em furos, entre o início do alerta e o limite de 80%
#             first_alert_hole = int(sub_monitor.iloc[first_alert_idx]["hole"])
#             drill_lead_time = max(0, idx_80 - first_alert_hole)
#         else:
#             drill_lead_time = 0
#
#         lead_times.append(drill_lead_time)
#
#     mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
#
#
#     print(f"MODELO: {model_name} (Janela de Persistência = {persistence_window})")
#     print(f"F1-score:  {f1:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall:    {rec:.4f}")
#     print(f"Accuracy:  {acc:.4f}")
#     print(f"AUC:       {auc:.4f}" if not np.isnan(auc) else "AUC:       Indefinido")
#     print(f"Mean Failure-Aligned Lead-Time: {mean_lead_time:.2f} furos antes da região de 80%")
#     print("\nMatriz de Confusão (Linhas = Real, Colunas = Predito):")
#
#     # Salva dados da curva ROC
#     if len(fpr) > 0:
#         roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
#         roc_df.to_csv(os.path.join(RESULTS_DIR, f"roc_data_{model_name.lower().replace(' ', '_')}.csv"), index=False)
#
#     return {
#         "Model": model_name,
#         "F1-score": round(f1, 4),
#         "Precision": round(prec, 4),
#         "Recall": round(rec, 4),
#         "Accuracy": round(acc, 4),
#         "AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
#         "Mean Lead-Time (Holes)": round(mean_lead_time, 2),
#         "TN": cm[0, 0],
#         "FP": cm[0, 1],
#         "FN": cm[1, 0],
#         "TP": cm[1, 1],
#     }
#
# # EXECUÇÃO CONJUNTA (FUSÃO HÍBRIDA)
# if __name__ == "__main__":
#     models_config = [
#         {
#             # Note o nome do arquivo conforme gerado pelo script híbrido (results_LSTM_hybrid.csv)
#             "filename": "results_LSTM_hybrid_option_A.csv",
#             "model_name": "LSTM-AE (Hybrid)",
#             "score_col": "mse",
#             "persistence_window": 2,
#         },
#         {
#             "filename": "results_autoencoder_hybrid_option_A.csv",
#             "model_name": "MLP-AE (Hybrid)",
#             "score_col": "mse",
#             "persistence_window": 2,
#         },
#         {
#             "filename": "results_xgboost_hybrid_option_A.csv",
#             "model_name": "XGBoost (Hybrid)",
#             "score_col": "score",
#             "persistence_window": 2,
#         },
#     ]
#
#     summary_list = []
#     for cfg in models_config:
#         res = evaluate_single_model(
#             filename=cfg["filename"],
#             model_name=cfg["model_name"],
#             score_col=cfg["score_col"],
#             persistence_window=cfg["persistence_window"],
#         )
#         if res:
#             summary_list.append(res)
#
#     if summary_list:
#         summary_df = pd.DataFrame(summary_list)
#         print("TABELA COMPARATIVA CONSOLIDADA (HYBRID FUSION)")
#         print(summary_df.to_string(index=False))
#         out_summary = os.path.join(RESULTS_DIR, "summary_models_hybrid_comparison.csv")
#         summary_df.to_csv(out_summary, index=False)
#         print(f"\nTabela salva em: {out_summary}")

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import (
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     accuracy_score
# )
#
# # ============================================================
# # CONFIGURAÇÕES
# # ============================================================
#
# CALIBRATION_HOLES = 5
# MONITORING_START_HOLE = 6
# PERSISTENCE_W = 2
#
# SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
#
# PROJECT_ROOT = SCRIPT_PATH
#
# while not os.path.isdir(os.path.join(PROJECT_ROOT, "data")):
#     parent = os.path.dirname(PROJECT_ROOT)
#
#     if parent == PROJECT_ROOT:
#         raise FileNotFoundError(
#             "Não foi possível localizar a pasta 'data'."
#         )
#
#     PROJECT_ROOT = parent
#
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
# PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
#
# os.makedirs(PLOT_DIR, exist_ok=True)
#
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif"],
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42
# })
#
# MODELS = {
#     "LSTM-AE": {
#         "file": "results_LSTM_hybrid_option_A.csv",
#         "cmap": "Purples"
#     },
#     "MLP-AE": {
#         "file": "results_autoencoder_hybrid_option_A.csv",
#         "cmap": "Greens"
#     },
#     "XGBoost": {
#         "file": "results_xgboost_hybrid_option_A.csv",
#         "cmap": "Oranges"
#     }
# }
#
# # ============================================================
# # MATRIZ REGIONAL POR BROCA
# # ============================================================
#
# def calculate_regional_confusion_matrix(df):
#
#     df = df.copy()
#
#     required = ["drill", "hole", "label", "prediction"]
#
#     missing = [c for c in required if c not in df.columns]
#
#     if missing:
#         raise ValueError(f"Colunas ausentes: {missing}")
#
#     for col in ["hole", "label", "prediction"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
#     if df[["drill", "hole"]].isna().any().any():
#         raise ValueError("Identificadores de broca/furo inválidos.")
#
#     if df.duplicated(["drill", "hole"]).any():
#         raise ValueError("Existem pares broca/furo duplicados.")
#
#     # Exclui calibração individualmente para cada broca.
#     df = df[
#         df["hole"] >= MONITORING_START_HOLE
#         ].copy()
#
#     # Somente regiões normal (0) e final (1).
#     df = df[
#         df["label"].isin([0, 1])
#     ].copy()
#
#     if df.empty:
#         raise ValueError("Nenhum evento válido após os filtros.")
#
#     if df["prediction"].isna().any():
#         raise ValueError("Existem predições ausentes.")
#
#     if not df["prediction"].isin([0, 1]).all():
#         raise ValueError("As predições devem ser binárias (0/1).")
#
#     regional_results = []
#
#     for drill, sub in df.groupby("drill"):
#
#         sub = sub.sort_values("hole")
#
#         normal = sub[sub["label"] == 0]
#         failure = sub[sub["label"] == 1]
#
#         # Não inventar uma decisão para uma região ausente.
#         if normal.empty or failure.empty:
#             raise ValueError(
#                 f"Broca {drill}: região normal ou final ausente "
#                 "após o filtro de monitoramento."
#             )
#
#         # Decisão regional:
#         # 1 = pelo menos um alarme final na região.
#         normal_alert = int(
#             (normal["prediction"] == 1).any()
#         )
#
#         failure_alert = int(
#             (failure["prediction"] == 1).any()
#         )
#
#         regional_results.append({
#             "drill": drill,
#             "region": "Normal",
#             "y_true": 0,
#             "y_pred": normal_alert
#         })
#
#         regional_results.append({
#             "drill": drill,
#             "region": "Failure-aligned",
#             "y_true": 1,
#             "y_pred": failure_alert
#         })
#
#     regional_df = pd.DataFrame(regional_results)
#
#     cm = confusion_matrix(
#         regional_df["y_true"],
#         regional_df["y_pred"],
#         labels=[0, 1]
#     )
#
#     return cm, regional_df
#
#
# # ============================================================
# # DESENHO DA MATRIZ
# # ============================================================
#
# def draw_matrix(ax, cm, title, cmap):
#
#     ax.imshow(cm, cmap=cmap, vmin=0)
#
#     max_value = max(int(cm.max()), 1)
#
#     for i in range(2):
#         for j in range(2):
#
#             value = int(cm[i, j])
#
#             color = (
#                 "white"
#                 if value > 0.55 * max_value
#                 else "black"
#             )
#
#             ax.text(
#                 j,
#                 i,
#                 str(value),
#                 ha="center",
#                 va="center",
#                 fontsize=13,
#                 fontweight="bold",
#                 color=color
#             )
#
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#
#     ax.set_xticklabels(["No Alert", "Alert"])
#     ax.set_yticklabels(["Normal", "Failure-aligned"])
#
#     ax.set_xlabel("Predicted Label")
#     ax.set_ylabel("True Label")
#
#     ax.set_title(title, fontweight="bold")
#
#     ax.tick_params(length=0)
#
#
# # ============================================================
# # EXECUÇÃO
# # ============================================================
#
# def main():
#
#     fig, axes = plt.subplots(
#         1,
#         3,
#         figsize=(7.2, 2.8)
#     )
#
#     summary = []
#     regional_tables = []
#
#     for ax, (model_name, config) in zip(
#             axes, MODELS.items()
#     ):
#
#         filepath = os.path.join(
#             RESULTS_DIR,
#             config["file"]
#         )
#
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(filepath)
#
#         df = pd.read_csv(filepath)
#
#         cm, regional_df = calculate_regional_confusion_matrix(df)
#
#         regional_df.insert(0, "Model", model_name)
#         regional_tables.append(regional_df)
#
#         y_true = regional_df["y_true"]
#         y_pred = regional_df["y_pred"]
#
#         tn, fp, fn, tp = cm.ravel()
#
#         summary.append({
#             "Model": model_name,
#             "TN": int(tn),
#             "FP": int(fp),
#             "FN": int(fn),
#             "TP": int(tp),
#             "F1-score": f1_score(
#                 y_true, y_pred, zero_division=0
#             ),
#             "Precision": precision_score(
#                 y_true, y_pred, zero_division=0
#             ),
#             "Recall": recall_score(
#                 y_true, y_pred, zero_division=0
#             ),
#             "Accuracy": accuracy_score(
#                 y_true, y_pred
#             ),
#             "N drills": regional_df["drill"].nunique(),
#             "N regional decisions": len(regional_df)
#         })
#
#         draw_matrix(
#             ax,
#             cm,
#             model_name,
#             config["cmap"]
#         )
#
#         print(f"\n{model_name}")
#         print(cm)
#         print(
#             f"N drills: {regional_df['drill'].nunique()} | "
#             f"N decisions: {len(regional_df)}"
#         )
#
#     fig.tight_layout(w_pad=1.0)
#
#     output_base = os.path.join(
#         PLOT_DIR,
#         "regional_confusion_matrices_hybrid"
#     )
#
#     fig.savefig(
#         output_base + ".pdf",
#         bbox_inches="tight",
#         pad_inches=0.03
#     )
#
#     fig.savefig(
#         output_base + ".png",
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0.03
#     )
#
#     plt.show()
#     plt.close(fig)
#
#     summary_df = pd.DataFrame(summary)
#
#     summary_df.to_csv(
#         os.path.join(
#             RESULTS_DIR,
#             "regional_confusion_summary_hybrid.csv"
#         ),
#         index=False
#     )
#
#     pd.concat(
#         regional_tables,
#         ignore_index=True
#     ).to_csv(
#         os.path.join(
#             RESULTS_DIR,
#             "regional_decisions_by_drill_hybrid.csv"
#         ),
#         index=False
#     )
#
#     print("\nREGIONAL CONFUSION SUMMARY")
#     print(summary_df.to_string(index=False))
#
#
# if __name__ == "__main__":
#     main()
#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CALIBRATION_HOLES = 5
MONITORING_START_HOLE = 6
PERSISTENCE_W = 2

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = SCRIPT_PATH

while not os.path.isdir(os.path.join(PROJECT_ROOT, "data")):
    parent = os.path.dirname(PROJECT_ROOT)

    if parent == PROJECT_ROOT:
        raise FileNotFoundError(
            "Não foi possível localizar a pasta 'data'."
        )

    PROJECT_ROOT = parent

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

MODELS = {
    "LSTM-AE": {
        "file": "results_LSTM_hybrid_option_A.csv",
        "cmap": "Purples"
    },
    "MLP-AE": {
        "file": "results_autoencoder_hybrid_option_A.csv",
        "cmap": "Greens"
    },
    "XGBoost": {
        "file": "results_xgboost_hybrid_option_A.csv",
        "cmap": "Oranges"
    }
}

# ============================================================
# MATRIZ REGIONAL POR BROCA
# ============================================================

def calculate_regional_confusion_matrix(df):

    df = df.copy()

    required = ["drill", "hole", "label", "prediction"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    for col in ["hole", "label", "prediction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["drill", "hole"]].isna().any().any():
        raise ValueError("Identificadores de broca/furo inválidos.")

    if df.duplicated(["drill", "hole"]).any():
        raise ValueError("Existem pares broca/furo duplicados.")

    # Exclui calibração individualmente para cada broca.
    df = df[
        df["hole"] >= MONITORING_START_HOLE
        ].copy()

    # Somente regiões normal (0) e final (1).
    df = df[
        df["label"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError("Nenhum evento válido após os filtros.")

    if df["prediction"].isna().any():
        raise ValueError("Existem predições ausentes.")

    if not df["prediction"].isin([0, 1]).all():
        raise ValueError("As predições devem ser binárias (0/1).")

    regional_results = []

    for drill, sub in df.groupby("drill"):

        sub = sub.sort_values("hole")

        normal = sub[sub["label"] == 0]
        failure = sub[sub["label"] == 1]

        # Não inventar uma decisão para uma região ausente.
        if normal.empty or failure.empty:
            raise ValueError(
                f"Broca {drill}: região normal ou final ausente "
                "após o filtro de monitoramento."
            )

        # Decisão regional:
        # 1 = pelo menos um alarme final na região.
        normal_alert = int(
            (normal["prediction"] == 1).any()
        )

        failure_alert = int(
            (failure["prediction"] == 1).any()
        )

        regional_results.append({
            "drill": drill,
            "region": "Normal",
            "y_true": 0,
            "y_pred": normal_alert
        })

        regional_results.append({
            "drill": drill,
            "region": "Failure-aligned",
            "y_true": 1,
            "y_pred": failure_alert
        })

    regional_df = pd.DataFrame(regional_results)

    cm = confusion_matrix(
        regional_df["y_true"],
        regional_df["y_pred"],
        labels=[0, 1]
    )

    return cm, regional_df


# ============================================================
# DESENHO DA MATRIZ
# ============================================================

def draw_matrix(ax, cm, title, cmap):

    ax.imshow(cm, cmap=cmap, vmin=0)

    max_value = max(int(cm.max()), 1)

    for i in range(2):
        for j in range(2):

            value = int(cm[i, j])

            color = (
                "white"
                if value > 0.55 * max_value
                else "black"
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Alert", "Alert"])

    # Rótulos do eixo Y na vertical para economizar espaço e evitar sobreposição
    ax.set_yticklabels(
        ["Normal", "Failure-aligned"],
        rotation=90,
        va="center"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_title(title, fontweight="bold")

    ax.tick_params(length=0)


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    # Dimensões mais compactas e gridspec com espaçamento controlado
    fig = plt.figure(figsize=(4.5, 4.2))
    gs = fig.add_gridspec(2, 4, wspace=0.35, hspace=0.45)

    ax_lstm = fig.add_subplot(gs[0, :2])
    ax_mlp = fig.add_subplot(gs[0, 2:])
    ax_xgb = fig.add_subplot(gs[1, 1:3])

    axes = [ax_lstm, ax_mlp, ax_xgb]

    summary = []
    regional_tables = []

    for ax, (model_name, config) in zip(
            axes, MODELS.items()
    ):

        filepath = os.path.join(
            RESULTS_DIR,
            config["file"]
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        df = pd.read_csv(filepath)

        cm, regional_df = calculate_regional_confusion_matrix(df)

        regional_df.insert(0, "Model", model_name)
        regional_tables.append(regional_df)

        y_true = regional_df["y_true"]
        y_pred = regional_df["y_pred"]

        tn, fp, fn, tp = cm.ravel()

        summary.append({
            "Model": model_name,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "F1-score": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "Precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "Accuracy": accuracy_score(
                y_true, y_pred
            ),
            "N drills": regional_df["drill"].nunique(),
            "N regional decisions": len(regional_df)
        })

        draw_matrix(
            ax,
            cm,
            model_name,
            config["cmap"]
        )

        print(f"\n{model_name}")
        print(cm)
        print(
            f"N drills: {regional_df['drill'].nunique()} | "
            f"N decisions: {len(regional_df)}"
        )

    output_base = os.path.join(
        PLOT_DIR,
        "regional_confusion_matrices_hybrid"
    )

    fig.savefig(
        output_base + ".pdf",
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig.savefig(
        output_base + ".png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()
    plt.close(fig)

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "regional_confusion_summary_hybrid.csv"
        ),
        index=False
    )

    pd.concat(
        regional_tables,
        ignore_index=True
    ).to_csv(
        os.path.join(
            RESULTS_DIR,
            "regional_decisions_by_drill_hybrid.csv"
        ),
        index=False
    )

    print("\nREGIONAL CONFUSION SUMMARY")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()