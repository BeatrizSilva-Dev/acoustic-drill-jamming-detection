import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)

CALIBRATION_HOLES = 5
MONITORING_START_HOLE = CALIBRATION_HOLES + 1
PERSISTENCE_W = 2  

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH
while not os.path.isdir(os.path.join(PROJECT_ROOT, "data")):
    parent = os.path.dirname(PROJECT_ROOT)
    if parent == PROJECT_ROOT:
        raise FileNotFoundError("The 'data' folder could not be located.")
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
    "ps.fonttype": 42,
})

MODELS = {
    "LSTM-AE": {"file": "results_lstm_ultrasonic.csv", "cmap": "Purples"},
    "MLP-AE": {"file": "results_autoencoder_ultrasonic.csv", "cmap": "Greens"},
    "XGBoost": {"file": "results_xgboost_ultrasonic.csv", "cmap": "Oranges"},
}


def calculate_regional_confusion_matrix(df):
    df = df.copy()
    required = ["drill", "hole", "label", "prediction"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["hole", "label", "prediction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["drill", "hole"]].isna().any().any():
        raise ValueError("Invalid drill/hole identifiers.")
    if df.duplicated(["drill", "hole"]).any():
        raise ValueError("There are duplicate drill/hole pairs.")

    df = df[df["hole"] >= MONITORING_START_HOLE].copy()
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No valid events after filters.")
    if df["prediction"].isna().any():
        raise ValueError("There are missing predictions.")
    if not df["prediction"].isin([0, 1]).all():
        raise ValueError("The predictions must be binary (0/1).")

    regional_results = []
    for drill, sub in df.groupby("drill"):
        sub = sub.sort_values("hole")
        normal = sub[sub["label"] == 0]
        failure = sub[sub["label"] == 1]
        if normal.empty or failure.empty:
            raise ValueError(
                f"Drill {drill}: normal or failure-aligned region missing "
                "after the monitoring filter."
            )
        regional_results.append({
            "drill": drill, "region": "Normal", "y_true": 0,
            "y_pred": int((normal["prediction"] == 1).any()),
        })
        regional_results.append({
            "drill": drill, "region": "Failure-aligned", "y_true": 1,
            "y_pred": int((failure["prediction"] == 1).any()),
        })

    regional_df = pd.DataFrame(regional_results)
    cm = confusion_matrix(regional_df["y_true"], regional_df["y_pred"], labels=[0, 1])
    return cm, regional_df


def draw_matrix(ax, cm, title, cmap):
    ax.imshow(cm, cmap=cmap, vmin=0)
    max_value = max(int(cm.max()), 1)
    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            color = "white" if value > 0.55 * max_value else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Alert", "Alert"])
    ax.set_yticklabels(["Normal", "Failure-aligned"], rotation=90, va="center")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(length=0)


def main():
    fig = plt.figure(figsize=(4.5, 4.2))
    gs = fig.add_gridspec(2, 4, wspace=0.35, hspace=0.45)
    axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:]),
            fig.add_subplot(gs[1, 1:3])]
    summary = []
    regional_tables = []
    try:
        for ax, (model_name, config) in zip(axes, MODELS.items()):
            filepath = os.path.join(RESULTS_DIR, config["file"])
            if not os.path.isfile(filepath):
                raise FileNotFoundError(filepath)
            df = pd.read_csv(filepath)
            cm, regional_df = calculate_regional_confusion_matrix(df)
            regional_df.insert(0, "Model", model_name)
            regional_tables.append(regional_df)
            y_true, y_pred = regional_df["y_true"], regional_df["y_pred"]
            tn, fp, fn, tp = cm.ravel()
            summary.append({
                "Model": model_name,
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "F1-score": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "Accuracy": accuracy_score(y_true, y_pred),
                "N drills": regional_df["drill"].nunique(),
                "N regional decisions": len(regional_df),
            })
            draw_matrix(ax, cm, model_name, config["cmap"])
            print(f"\n{model_name}\n{cm}")
            print(f"N drills: {regional_df['drill'].nunique()} | "
                  f"N decisions: {len(regional_df)}")

        output_base = os.path.join(PLOT_DIR, "regional_confusion_matrices_ultrasonic")
        fig.savefig(output_base + ".pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output_base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(RESULTS_DIR,
                                       "regional_confusion_summary_ultrasonic.csv"), index=False)
        pd.concat(regional_tables, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "regional_decisions_by_drill_ultrasonic.csv"),
            index=False,
        )
        print("\nREGIONAL CONFUSION SUMMARY")
        print(summary_df.to_string(index=False))
        print(f"\nFigures saved to: {output_base}.pdf / .png")
        plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
