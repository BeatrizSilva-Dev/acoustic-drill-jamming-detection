#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera as Tabelas 2 e 3 sob a lógica revisada do protocolo.

Tabela 2 — desempenho dos modelos-base por domínio:
  - exclui os primeiros 5 furos de cada broca (calibração);
  - avalia apenas os rótulos 0 e 1 (rótulo -1 = transição fica excluído);
  - calcula F1, precisão, recall, acurácia e AUC no mesmo subconjunto de monitoramento;
  - calcula MLT com persistência W=2, contando apenas alertas iniciados a partir do furo 6
    e antes do limite de 80% da sequência de cada broca.

Tabela 3 — AUC dos modelos-base e das variantes de fine-tuning:
  - usa os CSVs ROC produzidos pelos scripts revisados (colunas fpr, tpr, threshold);
  - calcula AUC pela área sob a curva ROC;
  - as curvas ROC devem ter sido geradas a partir dos furos 6+ e dos rótulos 0/1.

Estrutura esperada:
  PROJECT_ROOT/data/results/

O script procura automaticamente a pasta data/results a partir do diretório deste arquivo
ou permite definir PROJECT_ROOT por variável de ambiente.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==============================
# CONFIGURAÇÃO DO PROTOCOLO
# ==============================
CALIBRATION_HOLES = 5
MONITORING_START_HOLE = CALIBRATION_HOLES + 1
PERSISTENCE_W = 2
FAILURE_BOUNDARY_FRACTION = 0.80

DOMAIN_ORDER = ["Common", "Ultrasonic", "Hybrid"]
MODEL_ORDER = ["LSTM-AE", "MLP-AE", "XGBoost"]
FT_VARIANTS = ["FT_LAST_LAYER", "FT_DECODER", "FT_ALL_LOW_LR"]

# CSVs produzidos pelos três scripts de modelo revisados.
BASE_RESULT_FILES = {
    "Common": {
        "MLP-AE": "results_autoencoder_common_option_A.csv",
        "LSTM-AE": "results_LSTM_common_option_A.csv",
        "XGBoost": "results_xgboost_common_option_A.csv",
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

# Nomes dos CSVs ROC gravados pelos scripts revisados.
# Cada arquivo deve conter fpr, tpr e threshold.
ROC_FILES = {
    "Common": {
        "MLP-AE": "common_mlp_roc_data_option_A.csv",
        "LSTM-AE": "common_lstm_roc_data_option_A.csv",
        "XGBoost": "common_xgboost_roc_data_option_A.csv",
        "MLP-AE — FT last layer": "common_mlp_ft_last_layer_roc_data_option_A.csv",
        "MLP-AE — FT decoder": "common_mlp_ft_decoder_roc_data_option_A.csv",
        "MLP-AE — FT all low LR": "common_mlp_ft_all_low_lr_roc_data_option_A.csv",
        "LSTM-AE — FT last layer": "common_lstm_ft_last_layer_roc_data_option_A.csv",
        "LSTM-AE — FT decoder": "common_lstm_ft_decoder_roc_data_option_A.csv",
        "LSTM-AE — FT all low LR": "common_lstm_ft_all_low_lr_roc_data_option_A.csv",
    },
    "Ultrasonic": {
        "MLP-AE": "ultrasonic_mlp_roc_data_option_A.csv",
        "LSTM-AE": "ultrasonic_lstm_roc_data_option_A.csv",
        "XGBoost": "ultrasonic_xgboost_roc_data_option_A.csv",
        "MLP-AE — FT last layer": "ultrasonic_mlp_ft_last_layer_roc_data_option_A.csv",
        "MLP-AE — FT decoder": "ultrasonic_mlp_ft_decoder_roc_data_option_A.csv",
        "MLP-AE — FT all low LR": "ultrasonic_mlp_ft_all_low_lr_roc_data_option_A.csv",
        "LSTM-AE — FT last layer": "ultrasonic_lstm_ft_last_layer_roc_data_option_A.csv",
        "LSTM-AE — FT decoder": "ultrasonic_lstm_ft_decoder_roc_data_option_A.csv",
        "LSTM-AE — FT all low LR": "ultrasonic_lstm_ft_all_low_lr_roc_data_option_A.csv",
    },
    "Hybrid": {
        "MLP-AE": "hybrid_mlp_roc_data_option_A.csv",
        "LSTM-AE": "hybrid_lstm_roc_data_option_A.csv",
        "XGBoost": "hybrid_xgboost_roc_data_option_A.csv",
        "MLP-AE — FT last layer": "hybrid_mlp_ft_last_layer_roc_data_option_A.csv",
        "MLP-AE — FT decoder": "hybrid_mlp_ft_decoder_roc_data_option_A.csv",
        "MLP-AE — FT all low LR": "hybrid_mlp_ft_all_low_lr_roc_data_option_A.csv",
        "LSTM-AE — FT last layer": "hybrid_lstm_ft_last_layer_roc_data_option_A.csv",
        "LSTM-AE — FT decoder": "hybrid_lstm_ft_decoder_roc_data_option_A.csv",
        "LSTM-AE — FT all low LR": "hybrid_lstm_ft_all_low_lr_roc_data_option_A.csv",
    },
}


def locate_project_root() -> Path:
    """Find PROJECT_ROOT containing data/results, or honor PROJECT_ROOT env var."""
    env_root = os.environ.get("PROJECT_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve().parent
    candidates.extend([here, *here.parents])

    for candidate in candidates:
        if (candidate / "data" / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a pasta 'data/results'. Coloque este script dentro do projeto "
        "ou defina a variável de ambiente PROJECT_ROOT com o caminho do projeto."
    )


PROJECT_ROOT = locate_project_root()
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def get_label_column(df: pd.DataFrame) -> str:
    for col in ("label", "label_real", "y_true"):
        if col in df.columns:
            return col
    raise KeyError("CSV sem coluna de rótulo: esperava 'label' ou 'label_real'.")


def get_hole_column(df: pd.DataFrame) -> str:
    for col in ("hole", "hole_id", "hole_number"):
        if col in df.columns:
            return col
    raise KeyError("CSV sem coluna do furo; necessária para excluir calibração e ordenar a sequência.")


def get_drill_column(df: pd.DataFrame) -> str:
    for col in ("drill", "drill_id", "tool", "tool_id"):
        if col in df.columns:
            return col
    raise KeyError("CSV sem coluna da broca; necessária para calcular MLT por broca.")


def get_score_column(df: pd.DataFrame, model: str) -> Optional[str]:
    """Use o score contínuo coerente com as ROC dos scripts revisados."""
    candidates = (
        ["score", "normalized_score", "mse", "anomaly_score"]
        if model == "XGBoost"
        else ["normalized_score", "mse", "score", "anomaly_score"]
    )
    for col in candidates:
        if col in df.columns:
            return col
    return None


def monitoring_binary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Filtra furos >= 6 e rótulos 0/1, sem misturar calibração na avaliação."""
    label_col = get_label_column(df)
    hole_col = get_hole_column(df)
    out = df.copy()
    out[hole_col] = pd.to_numeric(out[hole_col], errors="coerce")
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out = out.dropna(subset=[hole_col, label_col, "prediction"])
    out = out[out[hole_col] >= MONITORING_START_HOLE]
    out = out[out[label_col].isin([0, 1])].copy()
    out[label_col] = out[label_col].astype(int)
    out["prediction"] = out["prediction"].astype(int)
    return out, label_col, hole_col


def calculate_mlt_w2(df_full: pd.DataFrame) -> float:
    """Mean lead time in holes before the 80% boundary, using W=2 and holes 6+."""
    label_col = get_label_column(df_full)
    hole_col = get_hole_column(df_full)
    drill_col = get_drill_column(df_full)

    data = df_full.copy()
    data[hole_col] = pd.to_numeric(data[hole_col], errors="coerce")
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    data = data.dropna(subset=[hole_col, "prediction"])
    lead_times: list[float] = []

    for _, sub in data.groupby(drill_col, sort=True):
        sub = sub.sort_values(hole_col).reset_index(drop=True)
        n_holes = len(sub)
        if n_holes <= CALIBRATION_HOLES:
            continue

        # Boundary is based on the full physical sequence, not the shortened monitoring subset.
        boundary_hole = int(np.floor(FAILURE_BOUNDARY_FRACTION * n_holes))
        monitored = sub[sub[hole_col] >= MONITORING_START_HOLE].copy()
        monitored = monitored[monitored[hole_col] < boundary_hole]
        preds = monitored["prediction"].astype(int).to_numpy()
        holes = monitored[hole_col].astype(int).to_numpy()

        first_alert_hole = None
        for i in range(0, len(preds) - PERSISTENCE_W + 1):
            if np.all(preds[i:i + PERSISTENCE_W] == 1):
                first_alert_hole = int(holes[i])
                break

        lead_times.append(float(max(boundary_hole - first_alert_hole, 0)) if first_alert_hole is not None else 0.0)

    return float(np.mean(lead_times)) if lead_times else float("nan")


def calculate_table2_row(domain: str, model: str, filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV de resultados-base não encontrado: {path}")

    df_full = pd.read_csv(path)
    df_eval, label_col, _ = monitoring_binary_rows(df_full)
    if df_eval.empty:
        raise ValueError(f"Nenhuma observação válida de monitoramento em {path.name}.")

    y_true = df_eval[label_col].astype(int).to_numpy()
    y_pred = df_eval["prediction"].astype(int).to_numpy()

    score_col = get_score_column(df_eval, model)
    auc_value = np.nan
    if score_col is not None:
        auc_df = df_eval.dropna(subset=[score_col])
        if auc_df[label_col].nunique() == 2:
            auc_value = float(
                roc_auc_score(
                    auc_df[label_col].astype(int).to_numpy(),
                    pd.to_numeric(auc_df[score_col], errors="coerce").to_numpy(dtype=float),
                )
            )

    return {
        "Configuration": domain,
        "Model": model,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "MLT (Holes, W=2)": calculate_mlt_w2(df_full),
        "N monitoring": len(df_eval),
        "N drills": df_eval[get_drill_column(df_eval)].nunique(),
    }


def read_roc_auc(path: Path) -> float:
    """Read a revised ROC curve CSV (fpr/tpr columns) and integrate its area."""
    if not path.exists():
        raise FileNotFoundError(f"CSV ROC não encontrado: {path}")
    df = pd.read_csv(path)
    if not {"fpr", "tpr"}.issubset(df.columns):
        raise ValueError(
            f"{path.name} não parece ser um CSV ROC do script revisado: "
            "esperava as colunas 'fpr' e 'tpr'."
        )
    df = df.dropna(subset=["fpr", "tpr"]).copy()
    df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
    df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")
    df = df.dropna(subset=["fpr", "tpr"]).sort_values("fpr")
    if len(df) < 2:
        return float("nan")
    return float(np.trapezoid(df["tpr"].to_numpy(), df["fpr"].to_numpy()))


def build_table2() -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        for model in MODEL_ORDER:
            filename = BASE_RESULT_FILES[domain][model]
            rows.append(calculate_table2_row(domain, model, filename))

    table = pd.DataFrame(rows)
    table["Configuration"] = pd.Categorical(table["Configuration"], DOMAIN_ORDER, ordered=True)
    table["Model"] = pd.Categorical(table["Model"], MODEL_ORDER, ordered=True)
    return table.sort_values(["Configuration", "Model"]).reset_index(drop=True)


def build_table3() -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        for model_label, filename in ROC_FILES[domain].items():
            path = RESULTS_DIR / filename
            try:
                auc_value = read_roc_auc(path)
                status = "OK"
            except FileNotFoundError:
                # Missing FT output should not silently disappear; keep a row with NaN.
                auc_value = np.nan
                status = "FILE NOT FOUND"
            rows.append({
                "Configuration": domain,
                "Model / fine-tuning": model_label,
                "AUC": auc_value,
                "Status": status,
                "ROC file": filename,
            })
    table = pd.DataFrame(rows)
    table["Configuration"] = pd.Categorical(table["Configuration"], DOMAIN_ORDER, ordered=True)
    return table.sort_values(["Configuration", "Model / fine-tuning"]).reset_index(drop=True)


def save_table(df: pd.DataFrame, stem: str, numeric_cols: list[str], column_format: str) -> None:
    csv_path = RESULTS_DIR / f"{stem}.csv"
    tex_path = RESULTS_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)

    latex_df = df.copy()
    for col in numeric_cols:
        if col in latex_df.columns:
            latex_df[col] = latex_df[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(float(x)) else "--"
            )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=True, column_format=column_format))

    print(f"Salvo: {csv_path}")
    print(f"Salvo: {tex_path}")


def main() -> None:
    print(f"Raiz do projeto: {PROJECT_ROOT}")
    print(f"Diretório de resultados: {RESULTS_DIR}")
    print(
        f"Protocolo: calibração=furos 1–{CALIBRATION_HOLES}; "
        f"monitoramento a partir do furo {MONITORING_START_HOLE}; "
        f"transição -1 excluída das métricas binárias; MLT W={PERSISTENCE_W}."
    )

    table2 = build_table2()
    print("\n" + "=" * 110)
    print("TABELA 2 — DESEMPENHO DOS MODELOS-BASE NO PERÍODO DE MONITORAMENTO")
    print("=" * 110)
    print(table2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_table(
        table2,
        "table_2_model_performance_revised",
        ["F1-score", "Precision", "Recall", "Accuracy", "AUC", "MLT (Holes, W=2)"],
        "llcccccccccc",
    )

    table3 = build_table3()
    print("\n" + "=" * 110)
    print("TABELA 3 — AUC ROC DOS MODELOS-BASE E DAS VARIANTES DE FINE-TUNING")
    print("=" * 110)
    print(table3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_table(
        table3,
        "table_3_finetuning_auc_revised",
        ["AUC"],
        "llccc",
    )

    print("\nConcluído. Verifique linhas com 'FILE NOT FOUND' na Tabela 3; elas indicam CSVs ROC ausentes.")


if __name__ == "__main__":
    main()
