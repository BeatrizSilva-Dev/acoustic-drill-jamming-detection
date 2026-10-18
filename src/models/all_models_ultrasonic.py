import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# RESOLUÇÃO DINÂMICA DE DIRETÓRIOS VIA PACOTE OS
# ============================================================
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_PATH, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# ============================================================
# FUNÇÃO: LOCALIZAR O PRIMEIRO ALARME PERSISTENTE
# ============================================================
def find_persistent_alert(predictions, window=3):
    """
    Retorna o índice do primeiro furo onde ocorrem 'window' alarmes consecutivos.
    Evita que um ruído/falso positivo isolado seja contado como lead-time.
    """
    predictions = np.asarray(predictions).astype(int)
    for i in range(len(predictions) - window + 1):
        if np.all(predictions[i : i + window] == 1):
            return i
    return None

# ============================================================
# FUNÇÃO DE AVALIAÇÃO INDIVIDUAL DE CADA MODELO
# ============================================================
def evaluate_single_model(filename, model_name, score_col, persistence_window=4):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[AVISO] Arquivo não encontrado: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # 1. Filtro do protocolo 50/30/20: exclusão estrita da zona de transição (-1)
    df_eval = df[df["label"] != -1].copy()
    if df_eval.empty:
        raise ValueError(f"Sem amostras válidas após exclusão da transição para {model_name}.")

    y_true = df_eval["label"].astype(int).values
    y_pred = df_eval["prediction"].astype(int).values

    # 2. Métricas de classificação
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 3. Tratamento robusto para ROC/AUC (remove NaNs gerados pelo SEQ_LEN inicial)
    y_score_raw = df_eval[score_col].astype(float).values
    valid_mask = np.isfinite(y_score_raw)

    y_true_auc = y_true[valid_mask]
    y_score_auc = y_score_raw[valid_mask]

    if len(np.unique(y_true_auc)) < 2:
        auc = np.nan
        fpr, tpr = np.array([]), np.array([])
    else:
        auc = roc_auc_score(y_true_auc, y_score_auc)
        fpr, tpr, _ = roc_curve(y_true_auc, y_score_auc)

    # 4. Lead-time corrigido: apenas alarmes persistentes antes da marca de 80%
    lead_times = []

    for drill in df["drill"].unique():

        sub_drill = (
            df[df["drill"] == drill]
            .sort_values("hole")
            .reset_index(drop=True)
        )

        n_holes = len(sub_drill)
        idx_80 = int(np.floor(0.80 * n_holes))

        # Only consider persistent alarms occurring before
        # the beginning of the failure-aligned anomalous region.
        preds_pre80 = (
            sub_drill["prediction"]
            .astype(int)
            .values[:idx_80]
        )

        first_alert_idx = find_persistent_alert(
            preds_pre80,
            window=persistence_window
        )

        if first_alert_idx is not None:
            drill_lead_time = idx_80 - first_alert_idx
        else:
            drill_lead_time = 0

        lead_times.append(drill_lead_time)

    mean_lead_time = np.mean(lead_times)

    # Exibição individual detalhada
    print("=" * 65)
    print(f"MODELO: {model_name} (Janela de Persistência = {persistence_window})")
    print("=" * 65)
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}" if not np.isnan(auc) else "AUC:       Indefinido")
    print(
        f"Mean Failure-Aligned Lead-Time: "
        f"{mean_lead_time:.2f} furos antes do início da região de 80%"
    )
    print("\nMatriz de Confusão (Linhas = Real, Colunas = Predito):")
    print(cm)
    print()

    # Salva dados da curva ROC para gráficos
    if len(fpr) > 0:
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(os.path.join(RESULTS_DIR, f"roc_data_{model_name.lower()}.csv"), index=False)

    return {
        "Model": model_name,
        "F1-score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "Mean Lead-Time (Holes)": round(mean_lead_time, 2),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }

# ============================================================
# EXECUÇÃO CONJUNTA E COMPARAÇÃO
# ============================================================
if __name__ == "__main__":
    models_config = [
        {
            "filename": "results_lstm_ultrasonic.csv",
            "model_name": "LSTM-AE (Ultrasonic)",
            "score_col": "hybrid_mse",
            "persistence_window": 3,
        },
        {
            "filename": "results_autoencoder_ultrasonic.csv",
            "model_name": "MLP-AE (Ultrasonic)",
            "score_col": "hybrid_mse",
            "persistence_window": 3,
        },
        {
            "filename": "results_xgboost_ultrasonic.csv",
            "model_name": "XGBoost (Ultrasonic)",
            "score_col": "score",
            "persistence_window": 3,
        },
    ]

    summary_list = []

    for cfg in models_config:
        res = evaluate_single_model(
            filename=cfg["filename"],
            model_name=cfg["model_name"],
            score_col=cfg["score_col"],
            persistence_window=cfg["persistence_window"],
        )
        if res:
            summary_list.append(res)

    # Tabela consolidada comparativa
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        print("=" * 80)
        print("TABELA COMPARATIVA CONSOLIDADA (ULTRASONIC)")
        print("=" * 80)
        print(summary_df.to_string(index=False))

        # Salva resumo em CSV para o artigo
        summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_models_ultrasonic_comparison.csv"), index=False)
        print(f"\nTabela salva em: {os.path.join(RESULTS_DIR, 'summary_models_ultrasonic_comparison.csv')}")