from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

CALIBRATION_HOLES = 5
PERSISTENCE_W = 2
SHOW_FIGURES = False
CHECK_EXPECTED_COUNTS = True
EXPECTED_HYBRID = {
    "LSTM-AE": (285, 20, 56, 97),
    "MLP-AE": (294, 11, 100, 53),
    "XGBoost": (237, 68, 15, 138),
}

ROOT = Path(__file__).resolve().parent
while not (ROOT / "data").is_dir():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("The project root containing the 'data' folder was not found.")
    ROOT = ROOT.parent
RESULTS = ROOT / "data" / "results"
OUTPUT = ROOT / "plots" / "all_hybrid_combined_plots"
OUTPUT.mkdir(parents=True, exist_ok=True)
THRESHOLD_FILE = RESULTS / "thresholds_hybrid.csv"

MODELS = [
    dict(name="LSTM-AE", filename="results_LSTM_hybrid.csv",
         score="normalized_score", color="#7209b7", threshold= None),
    dict(name="MLP-AE", filename="results_autoencoder_hybrid.csv",
         score="normalized_score", color="#008000", threshold= None),
    dict(name="XGBoost", filename="results_xgboost_hybrid.csv",
         score="score", color="#d97706", threshold=0.5),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def read_model(cfg):
    path = RESULTS / cfg["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Final file missing: {path}")
    df = pd.read_csv(path)
    needed = ["drill", "hole", "label", "prediction", cfg["score"]]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; available: {df.columns.tolist()}")
    if df["drill"].isna().any():
        raise ValueError(f"{path.name}: 'drill' contains missing values")
    for col in ["hole", "label", "prediction", cfg["score"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["hole"].isna().any() or (df["hole"] < 1).any() or (df["hole"] % 1 != 0).any():
        raise ValueError(f"{path.name}: hole must contain whole physical positions >=1")
    df["hole"] = df["hole"].astype(int)
    if df.duplicated(["drill", "hole"]).any():
        raise ValueError(f"{path.name}: duplicate pairs (drill, hole)")
    if not df["label"].isin([-1, 0, 1]).all():
        raise ValueError(f"{path.name}: label must be -1, 0, or 1")
    monitored = df[df["hole"] > CALIBRATION_HOLES]
    if not monitored["prediction"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: The prediction must be binary for ALL monitored boreholes.")
    df = df.sort_values(["drill", "hole"]).reset_index(drop=True)
    return df

def read_threshold_table():
    if not THRESHOLD_FILE.is_file():
        return None
    table = pd.read_csv(THRESHOLD_FILE)
    if {"drill_test", "threshold_mlp", "threshold_lstm"}.issubset(table.columns):
        return table
    if {"model", "drill", "threshold"}.issubset(table.columns):
        return table
    warnings.warn(f"{THRESHOLD_FILE.name}: format not recognized; "
                  "thresholds from the individual CSVs will be prioritized.")
    return None


def threshold_for(cfg, sub, table, drill):
    if cfg["name"] == "XGBoost":
        return 0.5
    for field in ("threshold", "normalized_threshold", "threshold_normalized", "decision_threshold"):
        if field in sub.columns:
            vals = pd.to_numeric(sub[field], errors="coerce").dropna().unique()
            if len(vals) > 1 and not np.allclose(vals, vals[0], rtol=1e-8, atol=1e-12):
                raise ValueError(f"{cfg['name']} drill {drill}: {field} not consistent when folding")
            if len(vals):
                value = float(vals[0])
                if not np.isfinite(value):
                    raise ValueError("Non-finite threshold")
                return value
    if table is not None:
        if "drill_test" in table.columns:
            field = "threshold_lstm" if cfg["name"] == "LSTM-AE" else "threshold_mlp"
            found = table.loc[table["drill_test"].astype(str).str.strip().str.lower() ==
                              str(drill).strip().lower(), field]
        else:
            found = table.loc[(table["model"].astype(str) == cfg["name"]) &
                              (table["drill"].astype(str).str.strip().str.lower() ==
                               str(drill).strip().lower()), "threshold"]
        vals = pd.to_numeric(found, errors="coerce").dropna().unique()
        if len(vals) > 1 and not np.allclose(vals, vals[0], rtol=1e-8, atol=1e-12):
            raise ValueError(f"Conflicting thresholds for {cfg['name']} drill {drill}")
        if len(vals):
            return float(vals[0])
    warnings.warn(f"No threshold exported for {cfg['name']}, drill {drill}: "
                  "line omitted; it was NOT recalibrated during the test.")
    return None
RENAMED_TO_ORIGINAL = {
    f"drill_4mm_{new:02d}": f"drill_4mm_{new + 3:02d}"
    for new in range(1, 21)
}

ABSOLUTE_JAM_DRILLS = set()

SEGMENTED_ONLY = False

def locate_jams_file(drill):
    renamed = str(drill).strip().lower()
    if renamed not in RENAMED_TO_ORIGINAL:
        raise ValueError(f"Renamed drill bit not recognized: {drill!r}")
    original = RENAMED_TO_ORIGINAL[renamed]
    roots = [ROOT / "data" / "segmented"]
    if not SEGMENTED_ONLY:
        roots.append(ROOT / "data")
    candidates = set()
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for folder in search_root.rglob("*"):
            if not folder.is_dir():
                continue
            name = folder.name.lower()
            if name == original or name.startswith(original + "_"):
                path = folder / "jams.txt"
                if path.is_file():
                    candidates.add(path.resolve())
    candidates = sorted(candidates)
    if len(candidates) > 1:
        raise ValueError(
            f"More than one jams.txt for {renamed} (original folder {original}): "
            + "; ".join(map(str, candidates))
        )
    return candidates[0] if candidates else None, original


def read_jam_numbers(path):
    numbers = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for token in re.split(r"[,;\s]+", line):
            if not token:
                continue
            if not re.fullmatch(r"[0-9]+", token) or int(token) <= 0:
                raise ValueError(f"{path}, line {line_number}: invalid number {token!r}")
            numbers.append(int(token))
    if not numbers:
        raise ValueError(f"jams.txt empty: {path}")
    return np.asarray(numbers, dtype=np.int64)


def physical_jams_for_drill(drill, ref):
    renamed = str(drill).strip().lower()
    path, original = locate_jams_file(renamed)
    if path is not None:
        numbers = read_jam_numbers(path)
        if renamed in ABSOLUTE_JAM_DRILLS:
            positions = numbers
            if len(positions) > 1 and not np.all(np.diff(positions) > 0):
                raise ValueError(f"{path}: absolute positions must be increasing")
            interpretation = "absolute positions (without cumsum)"
        else:
            positions = np.cumsum(numbers)
            interpretation = "successive intervals (cumsum)"
        source = str(path)
    elif "is_jam" in ref.columns:
        flags = pd.to_numeric(ref["is_jam"], errors="coerce")
        if not flags.isin([0, 1]).all():
            raise ValueError(f"invalid is_jam for {renamed}: must be 0/1")
        positions = ref.loc[flags.eq(1), "hole"].to_numpy(dtype=int)
        source = "results_xgboost_hybrid.csv:is_jam"
        interpretation = "flags is_jam"
    else:
        raise FileNotFoundError(
            f"Could not find jams.txt for {renamed} (ORIGINAL folder {original}) "
            f"in data/segmented, nor an is_jam column in the CSV. "
            f"Example folder: data/segmented/{original}_batch_.../jams.txt"
        )
    positions = np.asarray(positions, dtype=int)
    observed = set(ref["hole"].astype(int))
    missing = [int(h) for h in positions if int(h) not in observed]
    if missing:
        raise ValueError(
            f"JAM(s) for {renamed} (original {original}) outside the CSV holes: {missing}. "
            f"Source: {source}; interpretation: {interpretation}. "
            "Check whether the file contains intervals or absolute positions, "
            "and whether the 'hole' column represents the complete physical numbering."
        )
    print(f"{renamed} -> {original}: JAMs {positions.tolist()} "
          f"[{interpretation}] (source: {source})")
    return positions, source


def shade_regions(ax, ref):
    ax.axvspan(0.5, CALIBRATION_HOLES + 0.5, color="gray", alpha=0.13,
               label="Calibration (holes 1–5)")
    for label, color, text in [(-1, "#eab308", "Transition (excluded from binary metrics)"),
                               (1, "#ef4444", "Failure-aligned region")]:
        selected = ref.loc[ref["label"] == label, "hole"].astype(int).to_numpy()
        if len(selected):
            for k, h in enumerate(selected):
                ax.axvspan(h - 0.5, h + 0.5, color=color, alpha=0.11,
                           label=text if k == 0 else "_nolegend_")
    ax.axvline(CALIBRATION_HOLES + 0.5, color="gray", ls="--", lw=0.8)

def plot_drill(drill, dfs, threshold_table):
    ref = dfs["XGBoost"].loc[dfs["XGBoost"]["drill"] == drill].sort_values("hole").copy()
    fig, ax1 = plt.subplots(figsize=(3.35, 3.35))
    fig.subplots_adjust(left=0.17, right=0.80, top=0.79, bottom=0.17)
    ax2 = ax1.twinx()
    shade_regions(ax1, ref)

    for cfg in MODELS:
        name = cfg["name"]
        sub = dfs[name].loc[dfs[name]["drill"] == drill].sort_values("hole").copy()
        if sub.empty:
            raise ValueError(f"Drill {drill} missing in {name}")
        ax = ax2 if name == "XGBoost" else ax1
        threshold = threshold_for(cfg, sub, threshold_table, drill)
        ax.plot(sub["hole"], sub[cfg["score"]], color=cfg["color"],
                lw=1.05, marker={"LSTM-AE":"o", "MLP-AE":"v", "XGBoost":"s"}[name],
                ms=2.3, linestyle={"LSTM-AE":"-", "MLP-AE":"--", "XGBoost":"-."}[name],
                label=f"{name} score")
        if threshold is not None:
            ax.axhline(threshold, color=cfg["color"], ls=":", lw=1.0,
                       label=f"{name} threshold ({threshold:.3g})")
        alarms = sub[(sub["hole"] > CALIBRATION_HOLES) & (sub["prediction"] == 1)]
        finite = alarms[np.isfinite(alarms[cfg["score"]])]
        if not finite.empty:
            ax.scatter(finite["hole"], finite[cfg["score"]],
                       marker="x", s=32, linewidths=1.25, color=cfg["color"],
                       zorder=6, label=f"{name} definitive alarm (W=2)")
        missing = alarms[~np.isfinite(alarms[cfg["score"]])]
        if not missing.empty:
            warnings.warn(f"{name}, drill {drill}: {len(missing)} alarms with a missing score; "
                          "marked on the bottom edge of the corresponding axis.")
            ax.scatter(missing["hole"], np.zeros(len(missing)) if name == "XGBoost"
                       else np.full(len(missing), ax.get_ylim()[0]),
                       marker="x", s=32, color=cfg["color"], zorder=6)

    ax1.set_xlabel("Hole number", fontsize=8, labelpad=2)
    ax1.set_ylabel("Normalized AE error", fontsize=7.4, labelpad=2)
    ax2.set_ylabel("XGBoost probability", fontsize=7.4, labelpad=2)
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0.5, int(ref["hole"].max()) + 0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.tick_params(axis="both", labelsize=7, pad=2)
    ax2.tick_params(axis="y", labelsize=7, pad=2)
    ax1.grid(axis="both", linestyle=":", alpha=0.25)

    jam_positions, jam_source = physical_jams_for_drill(drill, ref)
    if len(jam_positions):
        ax1.scatter(jam_positions, np.zeros(len(jam_positions)),
                    transform=ax1.get_xaxis_transform(), marker="x", s=55,
                    linewidths=1.8, color="red", clip_on=False, zorder=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    rename = {
        "Calibration (holes 1–5)": "Calibration",
        "Transition (excluded from binary metrics)": "Transition",
        "Failure-aligned region": "Failure-aligned",
        "LSTM-AE score": "LSTM-AE", "MLP-AE score": "MLP-AE",
        "XGBoost score": "XGBoost",
        "LSTM-AE definitive alarm (W=2)": "LSTM alarm",
        "MLP-AE definitive alarm (W=2)": "MLP alarm",
        "XGBoost definitive alarm (W=2)": "XGB alarm",
    }
    handles, labels, seen = [], [], set()
    for h, label in zip(h1 + h2, l1 + l2):
        if not label or label.startswith("_"):
            continue
        label = rename.get(label, label)
        label = re.sub(r" threshold \([^)]*\)", " threshold", label)
        if label not in seen:
            handles.append(h); labels.append(label); seen.add(label)
    handles.append(Line2D([0], [0], marker="x", color="red", linestyle="None",
                          markeredgewidth=1.6, markersize=6))
    labels.append("Jamming event")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, fontsize=5.3, frameon=False, columnspacing=0.55,
               handlelength=1.2, handletextpad=0.3, borderaxespad=0.1)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(drill))
    base = OUTPUT / f"drill_{safe}_hybrid_temporal"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return {"drill": drill, "n_holes": len(ref), "jam_holes": ";".join(map(str, jam_positions)), "jam_source": jam_source, **{
        f"alarms_{cfg['name']}": int(((dfs[cfg['name']]["drill"] == drill) &
                                      (dfs[cfg['name']]["hole"] > CALIBRATION_HOLES) &
                                      (dfs[cfg['name']]["prediction"] == 1)).sum())
        for cfg in MODELS}}


def main():
    dfs = {cfg["name"]: read_model(cfg) for cfg in MODELS}
    reference = dfs["XGBoost"]
    ref_keys = reference[["drill", "hole", "label"]].copy()
    ref_keys["drill"] = ref_keys["drill"].astype(str)
    ref_keys = set(map(tuple, ref_keys.to_numpy()))
    for name, df in dfs.items():
        keys = df[["drill", "hole", "label"]].copy()
        keys["drill"] = keys["drill"].astype(str)
        if set(map(tuple, keys.to_numpy())) != ref_keys:
            raise ValueError(f"The (drill, hole, label) pairs for {name} differ from those of XGBoost. "
                             "It is not safe to overlay curves with misaligned events.")
        evaluated = df[(df["hole"] > CALIBRATION_HOLES) & df["label"].isin([0, 1])]
        if CHECK_EXPECTED_COUNTS:
            yt = evaluated["label"].to_numpy(dtype=int)
            yp = evaluated["prediction"].to_numpy(dtype=int)
            counts = (int(((yt == 0) & (yp == 0)).sum()),
                      int(((yt == 0) & (yp == 1)).sum()),
                      int(((yt == 1) & (yp == 0)).sum()),
                      int(((yt == 1) & (yp == 1)).sum()))
            if counts != EXPECTED_HYBRID[name]:
                warnings.warn(f"{name}: TN/FP/FN/TP={counts}, different from Table 2 "
                              f"{EXPECTED_HYBRID[name]}. Check the prediction version/column.")
        print(f"{name}: {df['drill'].nunique()} drills; {len(evaluated)} monitored binary events")
    table = read_threshold_table()
    drills = reference["drill"].drop_duplicates().tolist()
    if len(drills) != 20:
        warnings.warn(f"{len(drills)} drill bits were found, not the 20 that were expected.")
    rows = [plot_drill(drill, dfs, table) for drill in drills]
    pd.DataFrame(rows).to_csv(OUTPUT / "temporal_plot_audit.csv", index=False)
    print(f"\n{len(rows)} figures generated in: {OUTPUT}")

if __name__ == "__main__":
    main()
