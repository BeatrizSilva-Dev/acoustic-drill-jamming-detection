import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_PATH = os.path.join(
    PROJ_ROOT,
    "data",
    "segmented",
    "drill_4mm_06_batch_00_collet_1_30-01-2025"
)

CHANNEL = "Tr4"

def find_audio(root_dir, hole_number, channel, folder_type):
    pattern = f"hole{str(hole_number).zfill(5)}.wav"

    for root, _, files in os.walk(root_dir):
        if folder_type in root.lower():
            for f in files:
                if pattern in f and channel in f:
                    return os.path.join(root, f)

    return None


files_to_plot = [
    find_audio(ROOT_PATH, 1, CHANNEL, "reg_mics"),       # (a) Audible - Normal
    find_audio(ROOT_PATH, 24, CHANNEL, "reg_mics"),      # (b) Audible - Wear
    find_audio(ROOT_PATH, 1, CHANNEL, "ultrasonic"),     # (c) Ultrasonic - Normal
    find_audio(ROOT_PATH, 24, CHANNEL, "ultrasonic")     # (d) Ultrasonic - Wear
]

if None in files_to_plot:
    print("ERROR: One or more audio files were not found. Check the folder names.")
else:
    # Fontes aumentadas globalmente para padrão de publicação nítida
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.labelweight': 'bold',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Proporção ligeiramente mais ampla para acomodar fontes maiores sem espremer
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    labels = [
        '(a) Audible - Hole 1',
        '(b) Audible - Hole 24',
        '(c) Ultrasonic - Hole 1',
        '(d) Ultrasonic - Hole 24'
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
            x_axis='time',
            y_axis='linear',
            ax=ax,
            cmap='magma',
            vmin=-80,
            vmax=0
        )

        # Título de cada subplot em negrito e maior
        ax.set_title(labels[i], fontsize=12, fontweight='bold', pad=7)

        if row == 0:  # Audible row
            ax.set_ylim(0, 20000)
            ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        else:         # Ultrasonic row
            ax.set_ylim(0, 48000)
            ax.set_ylabel('Frequency (Hz)', fontweight='bold')

        if col == 1:
            ax.set_ylabel('')

        if row == 0:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (s)', fontweight='bold')

    plt.subplots_adjust(
        left=0.12,
        right=0.86,
        top=0.92,
        bottom=0.12,
        hspace=0.32,
        wspace=0.20
    )

    cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.80])

    cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar.set_label('Intensity (dB)', weight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Salva em PDF vetorial (ideal para o Overleaf) e em PNG de alta resolução (600 DPI)
    plt.savefig("sensor_comparison_2x2.pdf", bbox_inches='tight', pad_inches=0.03)
    plt.savefig("sensor_comparison_2x2.png", dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close()