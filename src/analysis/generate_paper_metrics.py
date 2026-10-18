
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc as sklearn_auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from scipy.stats import norm, wilcoxon
except Exception:
    norm = None
    wilcoxon = None

CALIBRATION_HOLES = 5
MONITORING_START_HOLE = CALIBRATION_HOLES + 1
PERSISTENCE_W = 2
FAILURE_BOUNDARY_FRACTION = 0.80

DOMAIN_ORDER = ["Common", "Ultrasonic", "Hybrid"]
MODEL_ORDER = ["LSTM-AE", "MLP-AE", "XGBoost"]
FT_VARIANTS = ["FT_LAST_LAYER", "FT_DECODER", "FT_ALL_LOW_LR"]

BASE_RESULT_FILES = {
    "Common": {
        "MLP-AE": "results_autoencoder_common.csv",
        "LSTM-AE": "results_LSTM_common.csv",
        "XGBoost": "results_xgboost_common.csv",
    },
    "Ultrasonic": {
        "MLP-AE": "results_autoencoder_ultrasonic.csv",
        "LSTM-AE": "results_LSTM_ultrasonic.csv",
        "XGBoost": "results_xgboost_ultrasonic.csv",
    },
    "Hybrid": {
        "MLP-AE": "results_autoencoder_hybrid.csv",
        "LSTM-AE": "results_LSTM_hybrid.csv",
        "XGBoost": "results_xgboost_hybrid.csv",
    },
}

ROC_FILES = {
    "Common": {
        "MLP-AE": "common_mlp_roc_data.csv",
        "LSTM-AE": "common_lstm_roc_data.csv",
        "XGBoost": "common_xgboost_roc_data.csv",
        "MLP-AE — FT last layer": "common_mlp_ft_last_layer_roc_data.csv",
        "MLP-AE — FT decoder": "common_mlp_ft_decoder_roc_data.csv",
        "MLP-AE — FT all low LR": "common_mlp_ft_all_low_lr_roc_data.csv",
        "LSTM-AE — FT last layer": "common_lstm_ft_last_layer_roc_data.csv",
        "LSTM-AE — FT decoder": "common_lstm_ft_decoder_roc_data.csv",
        "LSTM-AE — FT all low LR": "common_lstm_ft_all_low_lr_roc_data.csv",
    },
    "Ultrasonic": {
        "MLP-AE": "ultrasonic_mlp_roc_data.csv",
        "LSTM-AE": "ultrasonic_lstm_roc_data.csv",
        "XGBoost": "ultrasonic_xgboost_roc_data.csv",
        "MLP-AE — FT last layer": "ultrasonic_mlp_ft_last_layer_roc_data.csv",
        "MLP-AE — FT decoder": "ultrasonic_mlp_ft_decoder_roc_data.csv",
        "MLP-AE — FT all low LR": "ultrasonic_mlp_ft_all_low_lr_roc_data.csv",
        "LSTM-AE — FT last layer": "ultrasonic_lstm_ft_last_layer_roc_data.csv",
        "LSTM-AE — FT decoder": "ultrasonic_lstm_ft_decoder_roc_data.csv",
        "LSTM-AE — FT all low LR": "ultrasonic_lstm_ft_all_low_lr_roc_data.csv",
    },
    "Hybrid": {
        "MLP-AE": "hybrid_mlp_roc_data.csv",
        "LSTM-AE": "hybrid_lstm_roc_data.csv",
        "XGBoost": "hybrid_xgboost_roc_data.csv",
        "MLP-AE — FT last layer": "hybrid_mlp_ft_last_layer_roc_data.csv",
        "MLP-AE — FT decoder": "hybrid_mlp_ft_decoder_roc_data.csv",
        "MLP-AE — FT all low LR": "hybrid_mlp_ft_all_low_lr_roc_data.csv",
        "LSTM-AE — FT last layer": "hybrid_lstm_ft_last_layer_roc_data.csv",
        "LSTM-AE — FT decoder": "hybrid_lstm_ft_decoder_roc_data.csv",
        "LSTM-AE — FT all low LR": "hybrid_lstm_ft_all_low_lr_roc_data.csv",
    },
}

FT_SCORE_FILES = {
    domain: f"results_MLP_LSTM_finetuning_3_variants_{domain.lower()}.csv"
    for domain in DOMAIN_ORDER
}


def locate_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    here = Path(__file__).resolve().parent
    candidates.extend([here, *here.parents])
    for candidate in candidates:
        if (candidate / "data" / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find 'data/results'. Place this script inside the project "
        "or set PROJECT_ROOT to the project's root folder."
    )

PROJECT_ROOT = locate_project_root()
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def first_existing_column(df: pd.DataFrame, options: tuple[str, ...], purpose: str) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise KeyError(f"CSV missing column for {purpose}; expected options: {options}")


def label_col(df: pd.DataFrame) -> str:
    return first_existing_column(df, ("label", "label_real", "y_true"), "label")


def position_col(df: pd.DataFrame) -> str:
    return first_existing_column(df, ("hole_index", "hole", "hole_id", "hole_number"), "hole")


def drill_col(df: pd.DataFrame) -> str:
    return first_existing_column(df, ("drill", "drill_id", "tool", "tool_id"), "drill bit identifier")


def score_col(df: pd.DataFrame, model: str) -> Optional[str]:
    candidates = (
        ("score", "normalized_score", "mse", "anomaly_score")
        if model == "XGBoost"
        else ("normalized_score", "mse", "score", "anomaly_score")
    )
    return next((c for c in candidates if c in df.columns), None)

def monitoring_binary_rows(df: pd.DataFrame, require_prediction: bool = True) -> tuple[pd.DataFrame, str, str, str]:
    ycol = label_col(df)
    pcol = position_col(df)
    dcol = drill_col(df)
    out = df.copy()
    out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    required = [pcol, ycol, dcol]
    if require_prediction:
        if "prediction" not in out.columns:
            raise KeyError("CSV without a 'prediction' column.")
        out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
        required.append("prediction")
    out = out.dropna(subset=required)
    out = out[(out[pcol] >= MONITORING_START_HOLE) & out[ycol].isin([0, 1])].copy()
    out[ycol] = out[ycol].astype(int)
    if require_prediction:
        out["prediction"] = out["prediction"].astype(int)
    return out, ycol, pcol, dcol

def calculate_mlt_w2(df_full: pd.DataFrame) -> float:
    ycol = label_col(df_full)
    pcol = position_col(df_full)
    dcol = drill_col(df_full)
    data = df_full.copy()
    data[pcol] = pd.to_numeric(data[pcol], errors="coerce")
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    data = data.dropna(subset=[dcol, pcol, "prediction"])
    lead_times: list[float] = []

    for _, sub in data.groupby(dcol, sort=True):
        sub = sub.sort_values(pcol).reset_index(drop=True)
        n_events = int(sub[pcol].max()) if not sub.empty else 0
        if n_events <= CALIBRATION_HOLES:
            continue
        boundary = int(np.floor(FAILURE_BOUNDARY_FRACTION * n_events))
        monitored = sub[(sub[pcol] >= MONITORING_START_HOLE) & (sub[pcol] < boundary)].copy()
        positions = monitored[pcol].astype(int).to_numpy()
        preds = monitored["prediction"].astype(int).to_numpy()
        first_alert_position: Optional[int] = None
        for i in range(0, len(preds) - PERSISTENCE_W + 1):
            consecutive_positions = np.all(np.diff(positions[i:i + PERSISTENCE_W]) == 1)
            persistent_positive = np.all(preds[i:i + PERSISTENCE_W] == 1)
            if consecutive_positions and persistent_positive:
                first_alert_position = int(positions[i])
                break
        lead_times.append(
            float(max(boundary - first_alert_position, 0))
            if first_alert_position is not None else 0.0
        )
    return float(np.mean(lead_times)) if lead_times else float("nan")


def calculate_table2_row(domain: str, model: str, filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Base results CSV not found: {path}")
    df_full = pd.read_csv(path)
    df_eval, ycol, _, dcol = monitoring_binary_rows(df_full, require_prediction=True)
    if df_eval.empty:
        raise ValueError(f"No valid monitoring observation in {path.name}.")

    y_true = df_eval[ycol].astype(int).to_numpy()
    y_pred = df_eval["prediction"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sc = score_col(df_eval, model)
    auc_value = float("nan")
    n_auc = 0
    if sc is not None:
        auc_df = df_eval.copy()
        auc_df[sc] = pd.to_numeric(auc_df[sc], errors="coerce")
        auc_df = auc_df.dropna(subset=[sc])
        n_auc = len(auc_df)
        if auc_df[ycol].nunique() == 2:
            auc_value = float(roc_auc_score(auc_df[ycol].astype(int), auc_df[sc].astype(float)))

    return {
        "Configuration": domain,
        "Model": model,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "MLT (holes; W=2)": calculate_mlt_w2(df_full),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N monitoring": int(len(df_eval)),
        "N AUC": int(n_auc),
        "N drills": int(df_eval[dcol].nunique()),
    }


def read_roc_auc(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"CSV ROC not found: {path}")
    df = pd.read_csv(path)
    if not {"fpr", "tpr"}.issubset(df.columns):
        raise ValueError(f"{path.name}: expected 'fpr' and 'tpr' columns.")
    df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
    df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")
    df = df.dropna(subset=["fpr", "tpr"])
    if len(df) < 2:
        return float("nan")
    return float(sklearn_auc(df["fpr"].to_numpy(), df["tpr"].to_numpy()))


def build_table2() -> pd.DataFrame:
    rows = [
        calculate_table2_row(domain, model, BASE_RESULT_FILES[domain][model])
        for domain in DOMAIN_ORDER for model in MODEL_ORDER
    ]
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
                value = read_roc_auc(path)
                status = "OK — regenerated with per-drill monitoring filter"
            except FileNotFoundError:
                value = np.nan
                status = "FILE NOT FOUND — rerun model script"
            except (ValueError, pd.errors.ParserError) as exc:
                value = np.nan
                status = f"INVALID ROC CSV — {exc}"
            rows.append({"Configuration": domain, "Model / fine-tuning": model_label,
                         "AUC": value, "Status": status, "ROC file": filename})
    table = pd.DataFrame(rows)
    table["Configuration"] = pd.Categorical(table["Configuration"], DOMAIN_ORDER, ordered=True)
    return table.sort_values(["Configuration", "Model / fine-tuning"]).reset_index(drop=True)


def per_drill_f1_from_df(df: pd.DataFrame, model: str, domain: str, variant: str) -> pd.DataFrame:
    eval_df, ycol, _, dcol = monitoring_binary_rows(df, require_prediction=True)
    rows = []
    for drill, sub in eval_df.groupby(dcol, sort=True):
        if sub[ycol].nunique() < 2:
            continue
        rows.append({"Configuration": domain, "Model": model, "Variant": variant,
                     "drill": drill, "F1-score": f1_score(sub[ycol], sub["prediction"], zero_division=0),
                     "N monitoring": len(sub)})
    return pd.DataFrame(rows)


def build_per_drill_metrics() -> pd.DataFrame:
    pieces = []
    for domain in DOMAIN_ORDER:
        for model in MODEL_ORDER:
            path = RESULTS_DIR / BASE_RESULT_FILES[domain][model]
            if not path.exists():
                print(f"[WARNING] No file for F1 via drill-down: {path.name}")
                continue
            df = pd.read_csv(path)
            pieces.append(per_drill_f1_from_df(df, model, domain, "BASE"))

        ft_path = RESULTS_DIR / FT_SCORE_FILES[domain]
        if not ft_path.exists():
            print(f"[WARNING] No fine-tuning scores by domain: {ft_path.name}")
            continue
        ft = pd.read_csv(ft_path)
        needed = {"model", "variant", "drill", "label", "prediction"}
        if not needed.issubset(ft.columns):
            print(f"[WARNING] {ft_path.name} does not have all the required columns: {sorted(needed)}")
            continue
        for model in ("MLP-AE", "LSTM-AE"):
            for variant in FT_VARIANTS:
                sub = ft[(ft["model"] == model) & (ft["variant"] == variant)].copy()
                if not sub.empty:
                    pieces.append(per_drill_f1_from_df(sub, model, domain, variant))
    pieces = [p for p in pieces if p is not None and not p.empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["Configuration", "Model", "Variant", "drill", "F1-score", "N monitoring"]
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p[idx]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    out = np.empty(n, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out.tolist()


def build_wilcoxon(per_drill: pd.DataFrame) -> pd.DataFrame:
    columns = ["Configuration", "Model", "Variant", "N paired drills", "Mean F1 base",
               "Mean F1 fine-tuned", "Mean delta F1", "Wilcoxon p", "Holm-adjusted p", "Effect size r"]
    if per_drill.empty or wilcoxon is None or norm is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for domain in DOMAIN_ORDER:
        domain_rows = []
        for model in ("MLP-AE", "LSTM-AE"):
            base = per_drill[(per_drill["Configuration"] == domain) &
                             (per_drill["Model"] == model) & (per_drill["Variant"] == "BASE")]
            for variant in FT_VARIANTS:
                ft = per_drill[(per_drill["Configuration"] == domain) &
                               (per_drill["Model"] == model) & (per_drill["Variant"] == variant)]
                paired = base[["drill", "F1-score"]].merge(
                    ft[["drill", "F1-score"]], on="drill", how="inner", suffixes=("_base", "_ft")
                )
                if len(paired) < 2:
                    continue
                delta = paired["F1-score_ft"].to_numpy() - paired["F1-score_base"].to_numpy()
                try:
                    test = wilcoxon(paired["F1-score_ft"], paired["F1-score_base"],
                                    alternative="two-sided", zero_method="wilcox", method="auto")
                    pval = float(test.pvalue)
                except Exception:
                    pval = float("nan")
                if np.isfinite(pval) and pval > 0:
                    z_approx = float(norm.isf(pval / 2.0))
                    effect_r = abs(z_approx) / np.sqrt(len(paired))
                elif pval == 0:
                    effect_r = float("nan")
                else:
                    effect_r = float("nan")
                domain_rows.append({
                    "Configuration": domain, "Model": model, "Variant": variant,
                    "N paired drills": int(len(paired)),
                    "Mean F1 base": float(paired["F1-score_base"].mean()),
                    "Mean F1 fine-tuned": float(paired["F1-score_ft"].mean()),
                    "Mean delta F1": float(np.mean(delta)), "Wilcoxon p": pval,
                    "Effect size r": float(effect_r),
                })
        if domain_rows:
            valid_idx = [i for i, r in enumerate(domain_rows) if np.isfinite(r["Wilcoxon p"])]
            adjusted = holm_adjust([domain_rows[i]["Wilcoxon p"] for i in valid_idx]) if valid_idx else []
            for i, p_adj in zip(valid_idx, adjusted):
                domain_rows[i]["Holm-adjusted p"] = p_adj
            for r in domain_rows:
                r.setdefault("Holm-adjusted p", float("nan"))
            rows.extend(domain_rows)
    return pd.DataFrame(rows, columns=columns)


def save_table(df: pd.DataFrame, stem: str, numeric_cols: list[str], column_format: str) -> None:
    csv_path = RESULTS_DIR / f"{stem}.csv"
    tex_path = RESULTS_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    latex_df = df.copy()
    for col in numeric_cols:
        if col in latex_df.columns:
            latex_df[col] = latex_df[col].map(
                lambda x: f"{float(x):.4f}" if pd.notna(x) and np.isfinite(float(x)) else "--"
            )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=True, column_format=column_format))
    print(f"Salvo: {csv_path}")
    print(f"Salvo: {tex_path}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Protocol: calibration positions 1–{CALIBRATION_HOLES}; monitoring from {MONITORING_START_HOLE}; "
          f"binary metrics labels 0/1 only; persistence W={PERSISTENCE_W}; MLT boundary={FAILURE_BOUNDARY_FRACTION:.0%}.")

    table2 = build_table2()
    print("\nBASE MODEL PERFORMANCE ON MONITORING EVENTS")
    print(table2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_table(table2, "table_2_model_performance_revised",
               ["F1-score", "Precision", "Recall", "Accuracy", "AUC", "MLT (holes; W=2)"],
               "llrrrrrrrrrrrrr")

    table3 = build_table3()
    print("\nROC AUC FOR BASE MODELS AND FINE-TUNING VARIANTS")
    print(table3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_table(table3, "table_3_finetuning_auc_revised", ["AUC"], "llrrl")

    per_drill = build_per_drill_metrics()
    per_drill_path = RESULTS_DIR / "per_drill_f1_revised.csv"
    per_drill.to_csv(per_drill_path, index=False)
    print(f"Salvo: {per_drill_path}")

    stats = build_wilcoxon(per_drill)
    stats_path = RESULTS_DIR / "wilcoxon_finetuning_vs_base_revised.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Salvo: {stats_path}")

    audit = RESULTS_DIR / "audit_protocol_revised.txt"
    with open(audit, "w", encoding="utf-8") as f:
        f.write("REVISED EVALUATION AUDIT\n")
        f.write(f"Calibration positions excluded from binary metrics and ROC: 1-{CALIBRATION_HOLES}\n")
        f.write(f"Monitoring starts at sequence position: {MONITORING_START_HOLE}\n")
        f.write("Binary metrics use only labels 0 and 1; transition label -1 is excluded.\n")
        f.write(f"Persistence window: W={PERSISTENCE_W}.\n")
        f.write(f"MLT boundary: floor({FAILURE_BOUNDARY_FRACTION:.2f} * sequence length); no confirmed alarm contributes 0.\n")
        f.write("MLT is an operational lead-to-boundary score, not remaining useful life or time to physical failure.\n")
        f.write("ROC CSVs must be regenerated by the *_ARTICLE_READY.py scripts; old ROC files are not valid after this correction.\n")
        f.write("AUC is calculated from continuous scores, not thresholded predictions.\n")
        f.write("Fine-tuning comparisons use paired per-drill F1 where both base and fine-tuned results are available.\n")
        f.write("Wilcoxon p-values are exploratory; Holm adjustment is applied within each sensing domain across available comparisons.\n")


if __name__ == "__main__":
    main()
