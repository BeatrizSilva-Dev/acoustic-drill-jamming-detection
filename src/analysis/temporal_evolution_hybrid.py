"""Evolução temporal Hybrid: resultados finais LODO, sem recalibrar o teste.

Dependências: numpy, pandas, matplotlib.
Execute a partir de qualquer pasta; o script procura a raiz que contém data/results.

A coluna prediction DEVE ser o alarme definitivo já persistido (W=2).
Para desenhar os limiares dos AEs, exporte-os no CSV de resultados
(coluna threshold ou normalized_threshold, repetida nas linhas do fold) OU em
 data/results/thresholds_hybrid_option_A.csv com colunas drill_test, threshold_mlp, threshold_lstm.
Para os X vermelhos, associa a numeração renomeada dos CSVs à pasta
original em data/segmented, lê o jams.txt individual e converte intervalos
sucessivos em posições acumuladas. Exceções com posições absolutas são
configuráveis. Alternativamente, usa is_jam do CSV quando disponível.
Sem limiar exportado, a curva e os alarmes são desenhados, mas não uma
linha de limiar inventada. Não se recalcula percentil na broca de teste.
"""
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

CALIBRATION_HOLES = 5
PERSISTENCE_W = 2  # documentação: NÃO reaplicar ao campo prediction
SHOW_FIGURES = False
CHECK_EXPECTED_COUNTS = True  # apenas aviso se os CSVs não forem a versão da Tabela 2
EXPECTED_HYBRID = {
    "LSTM-AE": (285, 20, 56, 97),
    "MLP-AE": (294, 11, 100, 53),
    "XGBoost": (237, 68, 15, 138),
}

ROOT = Path(__file__).resolve().parent
while not (ROOT / "data").is_dir():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Não foi encontrada a raiz do projeto com a pasta data.")
    ROOT = ROOT.parent
RESULTS = ROOT / "data" / "results"
OUTPUT = ROOT / "plots" / "all_hybrid_combined_plots_option_A"
OUTPUT.mkdir(parents=True, exist_ok=True)
THRESHOLD_FILE = RESULTS / "thresholds_hybrid_option_A.csv"

MODELS = [
    dict(name="LSTM-AE", filename="results_LSTM_hybrid_option_A.csv",
         score="normalized_score", color="#7209b7", threshold= None),
    dict(name="MLP-AE", filename="results_autoencoder_hybrid_option_A.csv",
         score="normalized_score", color="#008000", threshold= None),
    dict(name="XGBoost", filename="results_xgboost_hybrid_option_A.csv",
         score="score", color="#d97706", threshold=0.5),
]
# Se seu pipeline usa outro nome para o score NORMALIZADO, altere apenas
# MODELS[*]['score']. Não substitua por MSE bruto com limiar normalizado.

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def read_model(cfg):
    path = RESULTS / cfg["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo final ausente: {path}")
    df = pd.read_csv(path)
    needed = ["drill", "hole", "label", "prediction", cfg["score"]]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: colunas ausentes {missing}; disponíveis: {df.columns.tolist()}")
    if df["drill"].isna().any():
        raise ValueError(f"{path.name}: drill contém valores ausentes")
    for col in ["hole", "label", "prediction", cfg["score"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["hole"].isna().any() or (df["hole"] < 1).any() or (df["hole"] % 1 != 0).any():
        raise ValueError(f"{path.name}: hole deve conter posições físicas inteiras >=1")
    df["hole"] = df["hole"].astype(int)
    if df.duplicated(["drill", "hole"]).any():
        raise ValueError(f"{path.name}: pares (drill, hole) duplicados")
    if not df["label"].isin([-1, 0, 1]).all():
        raise ValueError(f"{path.name}: label deve estar em -1, 0, 1")
    # Calibração pode ter prediction=NaN, pois não integra o monitoramento.
    monitored = df[df["hole"] > CALIBRATION_HOLES]
    if not monitored["prediction"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: prediction deve ser binária para TODOS os furos monitorados")
    df = df.sort_values(["drill", "hole"]).reset_index(drop=True)
    return df


def read_threshold_table():
    """Aceita o formato original (wide) ou model/drill/threshold (long)."""
    if not THRESHOLD_FILE.is_file():
        return None
    table = pd.read_csv(THRESHOLD_FILE)
    if {"drill_test", "threshold_mlp", "threshold_lstm"}.issubset(table.columns):
        return table
    if {"model", "drill", "threshold"}.issubset(table.columns):
        return table
    warnings.warn(f"{THRESHOLD_FILE.name}: formato não reconhecido; "
                  "serão priorizados os limiares dos CSVs individuais.")
    return None


def threshold_for(cfg, sub, table, drill):
    if cfg["name"] == "XGBoost":
        return 0.5
    # Prioridade: limiar já exportado junto ao score normalizado.
    for field in ("threshold", "normalized_threshold", "threshold_normalized", "decision_threshold"):
        if field in sub.columns:
            vals = pd.to_numeric(sub[field], errors="coerce").dropna().unique()
            if len(vals) > 1 and not np.allclose(vals, vals[0], rtol=1e-8, atol=1e-12):
                raise ValueError(f"{cfg['name']} broca {drill}: {field} não é constante no fold")
            if len(vals):
                value = float(vals[0])
                if not np.isfinite(value):
                    raise ValueError("Limiar não finito")
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
            raise ValueError(f"Limiares conflitantes para {cfg['name']} broca {drill}")
        if len(vals):
            return float(vals[0])
    warnings.warn(f"Sem limiar exportado para {cfg['name']}, broca {drill}: "
                  "linha omitida; NÃO foi recalibrada no teste.")
    return None


# Os CSVs foram renomeados: 01 -> pasta original 04, ..., 20 -> 23.
# Não modifica a identificação exibida nas figuras nem os CSVs.
RENAMED_TO_ORIGINAL = {
    f"drill_4mm_{new:02d}": f"drill_4mm_{new + 3:02d}"
    for new in range(1, 21)
}

# Preencha com a identificação RENOMEADA das brocas cujo jams.txt já
# contenha POSIÇÕES ABSOLUTAS dos furos (e não intervalos). Exemplo:
# ABSOLUTE_JAM_DRILLS = {"drill_4mm_10"}
# Confirme primeiro o conteúdo real do arquivo da broca 10.
ABSOLUTE_JAM_DRILLS = set()

# Se True, só usa a pasta data/segmented. Se False, também procura em data/.
SEGMENTED_ONLY = False


def locate_jams_file(drill):
    renamed = str(drill).strip().lower()
    if renamed not in RENAMED_TO_ORIGINAL:
        raise ValueError(f"Broca renomeada não reconhecida: {drill!r}")
    original = RENAMED_TO_ORIGINAL[renamed]
    roots = [ROOT / "data" / "segmented"]
    if not SEGMENTED_ONLY:
        roots.append(ROOT / "data")
    candidates = set()
    for search_root in roots:
        if not search_root.is_dir():
            continue
        # Pasta original tem sufixos de batch, collet e data; comparação
        # por prefixo + '_' impede colisão de 04 com 040, por exemplo.
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
            f"Mais de um jams.txt para {renamed} (pasta original {original}): "
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
                raise ValueError(f"{path}, linha {line_number}: número inválido {token!r}")
            numbers.append(int(token))
    if not numbers:
        raise ValueError(f"jams.txt vazio: {path}")
    return np.asarray(numbers, dtype=np.int64)


def physical_jams_for_drill(drill, ref):
    """Busca o arquivo da pasta ORIGINAL correspondente à broca RENOMEADA.

    Regra padrão: números do jams.txt são intervalos sucessivos e usam
    soma acumulada. Exceções declaradas em ABSOLUTE_JAM_DRILLS já são
    posições absolutas. Nunca infere JAM automaticamente do último furo.
    """
    renamed = str(drill).strip().lower()
    path, original = locate_jams_file(renamed)
    if path is not None:
        numbers = read_jam_numbers(path)
        if renamed in ABSOLUTE_JAM_DRILLS:
            positions = numbers
            if len(positions) > 1 and not np.all(np.diff(positions) > 0):
                raise ValueError(f"{path}: posições absolutas devem ser crescentes")
            interpretation = "posições absolutas (sem cumsum)"
        else:
            positions = np.cumsum(numbers)
            interpretation = "intervalos sucessivos (cumsum)"
        source = str(path)
    elif "is_jam" in ref.columns:
        flags = pd.to_numeric(ref["is_jam"], errors="coerce")
        if not flags.isin([0, 1]).all():
            raise ValueError(f"is_jam inválido para {renamed}: somente 0/1")
        positions = ref.loc[flags.eq(1), "hole"].to_numpy(dtype=int)
        source = "results_xgboost_hybrid_option_A.csv:is_jam"
        interpretation = "flags is_jam"
    else:
        raise FileNotFoundError(
            f"Não encontrei jams.txt para {renamed} (pasta ORIGINAL {original}) "
            f"em data/segmented, nem coluna is_jam no CSV. "
            f"Exemplo de pasta: data/segmented/{original}_batch_.../jams.txt"
        )
    positions = np.asarray(positions, dtype=int)
    observed = set(ref["hole"].astype(int))
    missing = [int(h) for h in positions if int(h) not in observed]
    if missing:
        raise ValueError(
            f"JAM(s) de {renamed} (original {original}) fora dos furos do CSV: {missing}. "
            f"Origem: {source}; interpretação: {interpretation}. "
            "Confira se o arquivo contém intervalos ou posições absolutas, "
            "e se a coluna hole representa a numeração física completa."
        )
    print(f"{renamed} -> {original}: JAMs {positions.tolist()} "
          f"[{interpretation}] (origem: {source})")
    return positions, source


def shade_regions(ax, ref):
    # Região de calibração é distinta dos rótulos operacionais.
    ax.axvspan(0.5, CALIBRATION_HOLES + 0.5, color="gray", alpha=0.13,
               label="Calibration (holes 1–5)")
    # Pintar por rótulo REAL evita inconsistências de arredondamento de 50/80%.
    for label, color, text in [(-1, "#eab308", "Transition (excluded from binary metrics)"),
                               (1, "#ef4444", "Failure-aligned region")]:
        selected = ref.loc[ref["label"] == label, "hole"].astype(int).to_numpy()
        if len(selected):
            # Rótulo apenas uma vez na legenda.
            for k, h in enumerate(selected):
                ax.axvspan(h - 0.5, h + 0.5, color=color, alpha=0.11,
                           label=text if k == 0 else "_nolegend_")
    ax.axvline(CALIBRATION_HOLES + 0.5, color="gray", ls="--", lw=0.8)


def plot_drill(drill, dfs, threshold_table):
    """Uma figura por broca: dois AEs no eixo esquerdo e XGBoost no direito."""
    ref = dfs["XGBoost"].loc[dfs["XGBoost"]["drill"] == drill].sort_values("hole").copy()
    fig, ax1 = plt.subplots(figsize=(3.35, 3.35))
    fig.subplots_adjust(left=0.17, right=0.80, top=0.79, bottom=0.17)
    ax2 = ax1.twinx()
    shade_regions(ax1, ref)

    # Escalas diferentes: ambos os AEs usam score NORMALIZADO (esquerda),
    # e XGBoost usa probabilidade (direita). Nenhum limiar é recalculado.
    for cfg in MODELS:
        name = cfg["name"]
        sub = dfs[name].loc[dfs[name]["drill"] == drill].sort_values("hole").copy()
        if sub.empty:
            raise ValueError(f"Broca {drill} ausente em {name}")
        ax = ax2 if name == "XGBoost" else ax1
        threshold = threshold_for(cfg, sub, threshold_table, drill)
        ax.plot(sub["hole"], sub[cfg["score"]], color=cfg["color"],
                lw=1.05, marker={"LSTM-AE":"o", "MLP-AE":"v", "XGBoost":"s"}[name],
                ms=2.3, linestyle={"LSTM-AE":"-", "MLP-AE":"--", "XGBoost":"-."}[name],
                label=f"{name} score")
        if threshold is not None:
            ax.axhline(threshold, color=cfg["color"], ls=":", lw=1.0,
                       label=f"{name} threshold ({threshold:.3g})")
        # Marcar as decisões FINAIS já persistidas (W=2), sem reaplicação.
        alarms = sub[(sub["hole"] > CALIBRATION_HOLES) & (sub["prediction"] == 1)]
        finite = alarms[np.isfinite(alarms[cfg["score"]])]
        if not finite.empty:
            ax.scatter(finite["hole"], finite[cfg["score"]],
                       marker="x", s=32, linewidths=1.25, color=cfg["color"],
                       zorder=6, label=f"{name} definitive alarm (W=2)")
        # NaN nos scores não deve esconder um alarme final exportado.
        missing = alarms[~np.isfinite(alarms[cfg["score"]])]
        if not missing.empty:
            warnings.warn(f"{name}, broca {drill}: {len(missing)} alarmes com score ausente; "
                          "marcados na borda inferior do eixo correspondente.")
            ax.scatter(missing["hole"], np.zeros(len(missing)) if name == "XGBoost"
                       else np.full(len(missing), ax.get_ylim()[0]),
                       marker="x", s=32, color=cfg["color"], zorder=6)

    ax1.set_xlabel("Hole number", fontsize=8, labelpad=2)
    #ax1.set_title(f"Drill {re.search(r'(d+)$', str(drill)).group(1) if re.search(r'(d+)$', str(drill)) else drill}", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax1.set_ylabel("Normalized AE error", fontsize=7.4, labelpad=2)
    ax2.set_ylabel("XGBoost probability", fontsize=7.4, labelpad=2)
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0.5, int(ref["hole"].max()) + 0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.tick_params(axis="both", labelsize=7, pad=2)
    ax2.tick_params(axis="y", labelsize=7, pad=2)
    ax1.grid(axis="both", linestyle=":", alpha=0.25)

    # Um X vermelho para CADA JAM físico desta broca, na base do gráfico.
    # Por padrão, jams.txt: 39, 10, 3 -> 39, 49, 52. Exceções configuráveis.
    jam_positions, jam_source = physical_jams_for_drill(drill, ref)
    if len(jam_positions):
        ax1.scatter(jam_positions, np.zeros(len(jam_positions)),
                    transform=ax1.get_xaxis_transform(), marker="x", s=55,
                    linewidths=1.8, color="red", clip_on=False, zorder=12)

    # Legenda compacta acima da área de desenho, sem cobrir os dados.
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
    base = OUTPUT / f"drill_{safe}_hybrid_temporal_option_A"
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
            raise ValueError(f"Os pares (broca, furo, rótulo) de {name} diferem dos do XGBoost. "
                             "Não é seguro sobrepor curvas com eventos desalinhados.")
        evaluated = df[(df["hole"] > CALIBRATION_HOLES) & df["label"].isin([0, 1])]
        if CHECK_EXPECTED_COUNTS:
            yt = evaluated["label"].to_numpy(dtype=int)
            yp = evaluated["prediction"].to_numpy(dtype=int)
            counts = (int(((yt == 0) & (yp == 0)).sum()),
                      int(((yt == 0) & (yp == 1)).sum()),
                      int(((yt == 1) & (yp == 0)).sum()),
                      int(((yt == 1) & (yp == 1)).sum()))
            if counts != EXPECTED_HYBRID[name]:
                warnings.warn(f"{name}: TN/FP/FN/TP={counts}, diferente da Tabela 2 "
                              f"{EXPECTED_HYBRID[name]}. Verifique versão/coluna prediction.")
        print(f"{name}: {df['drill'].nunique()} brocas; {len(evaluated)} eventos binários monitorados")
    table = read_threshold_table()
    drills = reference["drill"].drop_duplicates().tolist()
    if len(drills) != 20:
        warnings.warn(f"Foram encontradas {len(drills)} brocas, não as 20 esperadas.")
    rows = [plot_drill(drill, dfs, table) for drill in drills]
    pd.DataFrame(rows).to_csv(OUTPUT / "temporal_plot_audit.csv", index=False)
    print(f"\n{len(rows)} figuras geradas em: {OUTPUT}")
    print("Cada broca: UMA figura combinada (PDF vetorial e PNG 600 dpi). CSV de auditoria salvo.")


if __name__ == "__main__":
    main()
