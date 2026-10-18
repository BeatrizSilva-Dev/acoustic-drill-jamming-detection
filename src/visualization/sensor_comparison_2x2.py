
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)

ROOT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "segmented",
    "drill_4mm_06_batch_00_collet_1_30-01-2025"
)

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

CHANNEL = "Tr4"


def find_audio(root_dir, hole_number, channel, folder_type):

    pattern = f"hole{hole_number:05d}.wav"

    for root, _, files in os.walk(root_dir):

        if folder_type.lower() in root.lower():

            for filename in files:

                if pattern in filename and channel in filename:
                    return os.path.join(root, filename)

    return None


files_to_plot = [
    find_audio(ROOT_PATH, 1, CHANNEL, "reg_mics"),
    find_audio(ROOT_PATH, 24, CHANNEL, "reg_mics"),
    find_audio(ROOT_PATH, 1, CHANNEL, "ultrasonic"),
    find_audio(ROOT_PATH, 24, CHANNEL, "ultrasonic")
]

if any(path is None for path in files_to_plot):

    print("ERROR: One or more audio files were not found.")

    for path in files_to_plot:
        print(path)

else:

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.2, 5.2)
    )

    labels = [
        "(a) Audible - Hole 1",
        "(b) Audible - Hole 24",
        "(c) Ultrasonic - Hole 1",
        "(d) Ultrasonic - Hole 24"
    ]

    img = None

    for i, path in enumerate(files_to_plot):

        row = i // 2
        col = i % 2
        ax = axes[row, col]

        y, sr = librosa.load(path, sr=None)

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=4096)),
            ref=np.max
        )

        img = librosa.display.specshow(
            D,
            sr=sr,
            hop_length=1024,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            cmap="magma",
            vmin=-80,
            vmax=0
        )

        ax.set_title(
            labels[i],
            fontsize=12,
            fontweight="bold",
            pad=7
        )

        if row == 0:
            ax.set_ylim(0, 20000)
        else:
            ax.set_ylim(0, 48000)

        if col == 0:
            ax.set_ylabel(
                "Frequency (Hz)",
                fontweight="bold"
            )
        else:
            ax.set_ylabel("")

        if row == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(
                "Time (s)",
                fontweight="bold"
            )

    plt.subplots_adjust(
        left=0.12,
        right=0.86,
        top=0.92,
        bottom=0.12,
        hspace=0.32,
        wspace=0.20
    )

    cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.80])

    cbar = fig.colorbar(
        img,
        cax=cbar_ax,
        format="%+2.0f dB"
    )

    cbar.set_label(
        "Intensity (dB)",
        weight="bold",
        fontsize=12
    )

    cbar.ax.tick_params(labelsize=11)

    pdf_path = os.path.join(
        PLOT_DIR,
        "sensor_comparison_2x2.pdf"
    )

    png_path = os.path.join(
        PLOT_DIR,
        "sensor_comparison_2x2.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.close(fig)

    print("Figures successfully saved:")
    print(pdf_path)
    print(png_path)
