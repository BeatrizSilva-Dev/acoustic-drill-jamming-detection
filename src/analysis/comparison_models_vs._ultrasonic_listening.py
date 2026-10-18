import os
import re
import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score, roc_auc_score,
)

CALIBRATION_HOLES = 5
MONITORING_START_HOLE = CALIBRATION_HOLES + 1

FT_EPOCHS = 10
FT_BATCH_SIZE = 16

FT_LR_LAST_LAYER = 1e-3
FT_LR_DECODER = 1e-3
FT_LR_ALL = 1e-4

FT_VARIANTS = [
    "FT_LAST_LAYER",
    "FT_DECODER",
    "FT_ALL_LOW_LR",
]

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

tf.get_logger().setLevel("ERROR")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH

while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    parent = os.path.dirname(PROJECT_ROOT)
    if parent == PROJECT_ROOT:
        raise FileNotFoundError(
            "Unable to locate the root directory containing the 'data' folder."
        )
    PROJECT_ROOT = parent

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ROOT_DATASET = os.path.join(DATA_DIR, "segmented")
LISTENING_FILE = os.path.join(
    DATA_DIR, "metadata", "manual_listening_complete_project.csv"
)
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MIC_A = "ultrasonic_mics"
MIC_B = None
TARGET_CHANNEL = "4"
N_MFCC = 20
SEQ_LEN = 5
CALIBRATION_HOLES = 5
EVAL_START_HOLE_INDEX = CALIBRATION_HOLES
PERSISTENCE_W = 2
THRESHOLD_PERCENTILE = 99.5
EPS = 1e-8

XGB_THRESHOLD = 0.5
XGB_SCALE_POS_WEIGHT = 8

SENSOR_DOMAIN = "ultrasonic_audio"

if not os.path.exists(LISTENING_FILE):
    raise FileNotFoundError(f"File not found: {LISTENING_FILE}")

def standardize_drill_folder_name(name):
    match = re.search(r"drill_4mm_(\d+)", str(name).lower())
    if match:
        num = int(match.group(1))
        if num >= 4:
            num -= 3
        return f"drill_4mm_{num:02d}"
    return str(name).lower().strip()


df_listening = pd.read_csv(LISTENING_FILE)
df_listening["drill"] = (
    df_listening["drill"].astype(str).str.lower().str.strip()
)

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)[0]
    return np.hstack(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(rms),
            np.std(rms),
        ]
    )
def fine_tune_autoencoder(model, X_adaptation, variant, model_type):
    if X_adaptation is None or len(X_adaptation) == 0:
        return model

    for layer in model.layers:
        layer.trainable = False

    if variant == "FT_LAST_LAYER":
        model.layers[-1].trainable = True
        lr = FT_LR_LAST_LAYER

    elif variant == "FT_DECODER":
        found = False

        for layer in model.layers:
            name = layer.name.lower()

            if (
                "decoder" in name
                or "output" in name
            ):
                layer.trainable = True
                found = True

        if not found:
            model.layers[-1].trainable = True

        lr = FT_LR_DECODER

    elif variant == "FT_ALL_LOW_LR":
        for layer in model.layers:
            layer.trainable = True

        lr = FT_LR_ALL

    else:
        raise ValueError(f"Unknown fine-tuning variant: {variant}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="mse"
    )

    if model_type.upper() == "LSTM":
        X_seq = create_padded_causal_sequences(
            X_adaptation,
            SEQ_LEN
        )

        model.fit(
            X_seq,
            X_seq,
            epochs=FT_EPOCHS,
            batch_size=min(FT_BATCH_SIZE, len(X_seq)),
            shuffle=False,
            verbose=0
        )

    else:
        model.fit(
            X_adaptation,
            X_adaptation,
            epochs=FT_EPOCHS,
            batch_size=min(FT_BATCH_SIZE, len(X_adaptation)),
            shuffle=False,
            verbose=0
        )

    return model

def create_sequences(X, seq_len):
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=float)
    return np.array([X[i : i + seq_len] for i in range(len(X) - seq_len + 1)])


def create_padded_causal_sequences(X, seq_len):
    X = np.asarray(X, dtype=float)

    if len(X) == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=float)

    sequences = []

    for end in range(len(X)):
        start = max(0, end - seq_len + 1)
        seq = X[start : end + 1]

        if len(seq) < seq_len:
            pad = np.repeat(seq[[0]], seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])

        sequences.append(seq)

    return np.asarray(sequences, dtype=float)


def create_lstm_autoencoder(n_features):
    inputs = Input(shape=(SEQ_LEN, n_features), name="input_sequence")

    encoded = LSTM(
        32,
        activation="relu",
        name="encoder_lstm",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(inputs)

    decoded = RepeatVector(SEQ_LEN, name="repeat_vector")(encoded)

    decoded = LSTM(
        32,
        activation="relu",
        name="decoder_lstm",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        return_sequences=True,
    )(decoded)

    outputs = TimeDistributed(
        Dense(
            n_features,
            name="output_dense",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        ),
        name="time_distributed_output",
    )(decoded)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def clone_lstm_with_weights(base_model, variant):
    clone = tf.keras.models.clone_model(base_model)
    clone.set_weights(base_model.get_weights())

    if variant == "FT_LAST_LAYER":
        clone.get_layer("encoder_lstm").trainable = False
        clone.get_layer("repeat_vector").trainable = False
        clone.get_layer("decoder_lstm").trainable = False
        clone.get_layer("time_distributed_output").trainable = True
        learning_rate = 1e-3

    elif variant == "FT_DECODER":
        clone.get_layer("encoder_lstm").trainable = False
        clone.get_layer("repeat_vector").trainable = False
        clone.get_layer("decoder_lstm").trainable = True
        clone.get_layer("time_distributed_output").trainable = True
        learning_rate = 1e-3

    elif variant == "FT_ALL_LOW_LR":
        for layer in clone.layers:
            layer.trainable = True
        learning_rate = 1e-4

    else:
        raise ValueError(f"Unknown fine-tuning variant: {variant}")

    clone.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return clone


def fine_tune_lstm_on_calibration(base_model, X_calibration_scaled, variant):
    model = clone_lstm_with_weights(base_model, variant)

    X_calib_seq = create_padded_causal_sequences(
        X_calibration_scaled,
        SEQ_LEN,
    )

    if len(X_calib_seq) == 0:
        raise ValueError("There are not enough sequences for fine-tuning.")

    model.fit(
        X_calib_seq,
        X_calib_seq,
        epochs=10,
        batch_size=min(5, len(X_calib_seq)),
        shuffle=False,
        verbose=0,
    )

    return model


def reconstruction_errors_mlp(model, X):
    recon = model.predict(X)
    return np.mean((X - recon) ** 2, axis=1)

def reconstruction_errors_lstm(model, X_seq):
    if len(X_seq) == 0:
        return np.array([])
    recon = model.predict(X_seq, verbose=0)
    return np.mean((X_seq - recon) ** 2, axis=(1, 2))


def map_lstm_sequence_errors_to_holes(errors_seq, n_holes):
    errors = np.full(n_holes, np.nan, dtype=float)
    if len(errors_seq) > 0:
        errors[: min(n_holes, len(errors_seq))] = errors_seq[:n_holes]
    return errors


def normalize_by_initial_baseline(errors, n_calibration=CALIBRATION_HOLES):
    calibration_errors = errors[:n_calibration]
    calibration_errors = calibration_errors[np.isfinite(calibration_errors)]

    if len(calibration_errors) == 0:
        return np.full_like(errors, np.nan, dtype=float), np.nan

    baseline = float(np.median(calibration_errors))
    normalized = errors / (baseline + EPS)
    return normalized, baseline


def persistence_alarm(
    raw_alarm,
    start_index=EVAL_START_HOLE_INDEX,
    window=PERSISTENCE_W,
):
    raw_alarm = np.asarray(raw_alarm, dtype=int)
    pred = np.zeros(len(raw_alarm), dtype=int)

    if len(raw_alarm) < start_index + window:
        return pred

    for i in range(
        start_index + window - 1,
        len(raw_alarm),
    ):
        current = raw_alarm[
            i - window + 1 : i + 1
        ]

        if np.all(current == 1):
            pred[i:] = 1
            break

    return pred


def build_training_data(drills_data, drill_test):
    X_train = []
    y_train = []

    for drill_name, data in drills_data.items():
        if drill_name == drill_test:
            continue

        mask = data["y"] != -1
        if np.any(mask):
            X_train.append(data["X"][mask])
            y_train.append(data["y"][mask])

    if not X_train:
        raise ValueError(f"No training data for {drill_test}.")

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train


def build_ae_normal_training_data(drills_data, drill_test):
    normal_by_drill = {}

    for drill_name, data in drills_data.items():
        if drill_name == drill_test:
            continue

        normal_mask = data["y"] == 0
        normal_X = data["X"][normal_mask]

        if len(normal_X) >= SEQ_LEN:
            normal_by_drill[drill_name] = normal_X

    if not normal_by_drill:
        raise ValueError(f"Without sufficient normal data for AE in the sludge of {drill_test}.")

    X_normal = np.vstack(list(normal_by_drill.values()))
    return X_normal, normal_by_drill


def create_mlp_autoencoder(n_features):
    inputs = Input(shape=(n_features,), name="input_features")

    encoded = Dense(
        32,
        activation="relu",
        name="encoder_dense_1",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(inputs)
    encoded = Dense(
        16,
        activation="relu",
        name="encoder_dense_2",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 1),
    )(encoded)

    decoded = Dense(
        32,
        activation="relu",
        name="decoder_dense",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 2),
    )(encoded)
    outputs = Dense(
        n_features,
        activation="linear",
        name="output_dense",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 3),
    )(decoded)

    model = Model(inputs, outputs, name="mlp_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def train_mlp_autoencoder(X_normal_train, n_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal_train)

    model = create_mlp_autoencoder(n_features)
    model.fit(
        X_scaled,
        X_scaled,
        epochs=50,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return scaler, model


def clone_mlp_with_weights(base_model, variant):
    clone = tf.keras.models.clone_model(base_model)
    clone.set_weights(base_model.get_weights())

    if variant == "FT_LAST_LAYER":
        for layer in clone.layers:
            layer.trainable = layer.name == "output_dense"
        learning_rate = 1e-3

    elif variant == "FT_DECODER":
        for layer in clone.layers:
            layer.trainable = layer.name in {"decoder_dense", "output_dense"}
        learning_rate = 1e-3

    elif variant == "FT_ALL_LOW_LR":
        for layer in clone.layers:
            layer.trainable = True
        learning_rate = 1e-4

    else:
        raise ValueError(f"Unknown fine-tuning variant: {variant}")

    clone.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return clone


def fine_tune_mlp_on_calibration(base_model, X_calibration_scaled, variant):
    model = clone_mlp_with_weights(base_model, variant)

    X_calibration_scaled = np.asarray(X_calibration_scaled, dtype=float)
    if len(X_calibration_scaled) == 0:
        raise ValueError("There is not enough data for fine-tuning the MLP.")

    model.fit(
        X_calibration_scaled,
        X_calibration_scaled,
        epochs=10,
        batch_size=min(5, len(X_calibration_scaled)),
        shuffle=False,
        verbose=0,
    )
    return model


def train_lstm_autoencoder(normal_by_drill, scaler):
    sequences = []

    for drill_name in sorted(normal_by_drill):
        X = normal_by_drill[drill_name]
        X_scaled = scaler.transform(X)
        seq = create_sequences(X_scaled, SEQ_LEN)
        if len(seq) > 0:
            sequences.append(seq)

    if not sequences:
        raise ValueError("No LSTM sequence available for training.")

    X_seq_train = np.vstack(sequences)
    model = create_lstm_autoencoder(X_seq_train.shape[2])

    model.fit(
        X_seq_train,
        X_seq_train,
        epochs=50,
        batch_size=8,
        shuffle=False,
        verbose=0,
    )

    return model, X_seq_train


def compute_global_threshold_from_training_scores(
    model,
    scaler,
    normal_by_drill,
    model_type,
):
    all_normal_scores = []
    baseline_records = []

    for drill_name in sorted(normal_by_drill):
        X_normal = normal_by_drill[drill_name]

        if model_type == "mlp":
            X_scaled = scaler.transform(X_normal)
            errors = reconstruction_errors_mlp(model, X_scaled)

        elif model_type == "lstm":
            X_scaled = scaler.transform(X_normal)
            X_seq = create_padded_causal_sequences(
                X_scaled,
                SEQ_LEN,
            )
            errors_seq = reconstruction_errors_lstm(model, X_seq)
            errors = map_lstm_sequence_errors_to_holes(
                errors_seq,
                len(X_normal),
            )

        else:
            raise ValueError("model_type must be 'mlp' or 'lstm'.")

        normalized, baseline = normalize_by_initial_baseline(
            errors,
            n_calibration=CALIBRATION_HOLES,
        )

        valid_scores = normalized[np.isfinite(normalized)]

        if len(valid_scores) == 0 or not np.isfinite(baseline):
            continue

        all_normal_scores.extend(valid_scores.tolist())

        baseline_records.append(
            {
                "drill": drill_name,
                "baseline_error": baseline,
                "n_normal_scores": len(valid_scores),
            }
        )

    if len(all_normal_scores) == 0:
        raise ValueError(
            f"It was not possible to define a global threshold for {model_type}."
        )

    threshold = float(
        np.percentile(
            all_normal_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    return (
        threshold,
        pd.DataFrame(baseline_records),
        np.asarray(all_normal_scores),
    )

def diagnostic_persistence_scores(
        scores,
        threshold,
        y_true,
        start_index=EVAL_START_HOLE_INDEX,
        window=PERSISTENCE_W,
):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    raw_alarm = (scores > threshold).astype(int)

    valid_scores = scores[start_index:]
    valid_true = y_true[start_index:]
    valid_raw = raw_alarm[start_index:]

    max_run = 0
    current_run = 0

    for value in valid_raw:
        if value == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    raw_crossings = np.where(valid_raw == 1)[0]

    first_raw = (
        int(raw_crossings[0] + start_index + 1)
        if len(raw_crossings) > 0
        else None
    )

    first_persistent = None

    if len(valid_raw) >= window:
        for i in range(window - 1, len(valid_raw)):
            current = valid_raw[i - window + 1 : i + 1]

            if np.all(current == 1):
                first_persistent = int(i + start_index + 1)
                break

    failure_mask = valid_true == 1
    normal_mask = valid_true == 0
    raw_failure_alarms = np.sum(
        valid_raw[failure_mask] == 1
    )

    raw_normal_alarms = np.sum(
        valid_raw[normal_mask] == 1
    )

    print(
        f"    raw alarms in normal:  {int(raw_normal_alarms)}"
    )

    print(
        f"    raw alarms in failure: {int(raw_failure_alarms)}"
    )
    print(f"    threshold:              {threshold:.6f}")
    print(f"    raw alarms:             {int(valid_raw.sum())}")
    print(f"    max consecutive alarms: {max_run}")
    print(f"    first raw crossing:     {first_raw}")
    print(f"    first persistent:       {first_persistent}")

    if np.any(normal_mask):
        print(
            f"    normal max score:       "
            f"{np.max(valid_scores[normal_mask]):.6f}"
        )

    if np.any(failure_mask):
        print(
            f"    failure max score:      "
            f"{np.max(valid_scores[failure_mask]):.6f}"
        )

    if np.any(failure_mask):
        print(
            f"    failure mean score:     "
            f"{np.mean(valid_scores[failure_mask]):.6f}"
        )

    print(
        f"    scores > threshold:     "
        f"{int(valid_raw.sum())}/{len(valid_raw)}"
    )
def save_roc_data(df_scores, score_col, output_path):
    required = {"drill", "label", score_col}
    missing = required.difference(df_scores.columns)
    if missing:
        raise KeyError(
            f"Cannot create ROC for {os.path.basename(output_path)}; "
            f"missing columns: {sorted(missing)}"
        )
    position_col = "hole_index" if "hole_index" in df_scores.columns else "hole"
    if position_col not in df_scores.columns:
        raise KeyError(
            f"Cannot create ROC for {os.path.basename(output_path)}; "
            "missing both 'hole_index' and 'hole'."
        )

    df = df_scores[["drill", position_col, "label", score_col]].copy()
    df[position_col] = pd.to_numeric(df[position_col], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["drill", position_col, "label", score_col])
    df = df[
        (df[position_col] >= MONITORING_START_HOLE)
        & (df["label"].isin([0, 1]))
    ].copy()

    if df.empty or df["label"].nunique() < 2:
        print(
            f"[WARNING] ROC cannot be calculated: fewer than two classes "
            f"in {os.path.basename(output_path)} after filtering sequence "
            f"positions >= {MONITORING_START_HOLE} and labels 0/1."
        )
        return float("nan")

    y_true = df["label"].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": thresholds}
    ).to_csv(output_path, index=False)

    print(
        f"[ROC] {os.path.basename(output_path)} | AUC={roc_auc:.4f} "
        f"| N={len(df)} | drills={df['drill'].nunique()} "
        f"| positions>={MONITORING_START_HOLE}, labels=0/1"
    )
    return float(roc_auc)


def evaluate_binary(scores_or_predictions, labels, start_index=EVAL_START_HOLE_INDEX):
    y_true = np.asarray(labels)[start_index:]
    y_pred = np.asarray(scores_or_predictions)[start_index:]

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    mask = np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "n": 0,
        }

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n": len(y_true),
    }

drills_data = {}

for drill_folder in sorted(os.listdir(ROOT_DATASET)):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path):
        continue

    dict_a = {}

    for root, _, fs in os.walk(path):
        for f in fs:
            if not f.lower().endswith(".wav"):
                continue

            if f"ch{TARGET_CHANNEL}" not in f.lower() and f"tr{TARGET_CHANNEL}" not in f.lower():
                continue

            hole = extract_hole_number(f)
            if hole is None:
                continue

            if MIC_A in root.lower():
                dict_a[hole] = os.path.join(root, f)

    available_holes = sorted(dict_a.keys())

    if len(available_holes) == 0:
        continue

    X_drill = []
    hole_indices = []

    for hole in available_holes:
        try:
            y_a, sr_a = librosa.load(dict_a[hole], sr=None)
            feat_a = extract_features(y_a, sr_a)

            X_drill.append(feat_a)
            hole_indices.append(hole)
        except Exception as exc:
            print(f"[WARNING] Failure at hole {hole} in {drill_folder}: {exc}")

    n_total = len(X_drill)
    if n_total == 0:
        continue

    y_drill = []
    for i in range(n_total):
        life_fraction = i / n_total
        if life_fraction < 0.50:
            label = 0
        elif life_fraction < 0.80:
            label = -1
        else:
            label = 1
        y_drill.append(label)

    drill_key = standardize_drill_folder_name(drill_folder)
    drills_data[drill_key] = {
        "X": np.asarray(X_drill),
        "y": np.asarray(y_drill),
        "holes": hole_indices,
    }

if len(drills_data) < 2:
    raise ValueError("At least two drill bits are required for the LODO benchmark.")

eligibility_records = []

for drill_name in sorted(drills_data):
    n_holes = len(drills_data[drill_name]["X"])
    eligible = n_holes > CALIBRATION_HOLES
    eligibility_records.append(
        {
            "drill": drill_name,
            "n_holes": n_holes,
            "calibration_holes": min(n_holes, CALIBRATION_HOLES),
            "monitoring_holes": max(n_holes - CALIBRATION_HOLES, 0),
            "eligible_for_lodo": eligible,
            "reason": (
                "calibration + monitoring available"
                if eligible
                else "fewer than 6 holes; no post-calibration monitoring period"
            ),
        }
    )

df_eligibility = pd.DataFrame(eligibility_records)
eligible_drills = df_eligibility.loc[
    df_eligibility["eligible_for_lodo"], "drill"
].tolist()
ineligible_drills = df_eligibility.loc[
    ~df_eligibility["eligible_for_lodo"], "drill"
].tolist()


print(f"Total drills loaded:              {len(drills_data)}")
print(f"Eligible drills (N > 5):          {len(eligible_drills)}")
print(f"Ineligible drills (N <= 5):       {len(ineligible_drills)}")
print(f"Calibration holes:                {CALIBRATION_HOLES}")
print(f"Monitoring starts at:             hole {CALIBRATION_HOLES + 1}")
print("\nDetails:")
for row in eligibility_records:
    status = "OK" if row["eligible_for_lodo"] else "SKIP"
    print(
        f"  [{status}] {row['drill']}: "
        f"N={row['n_holes']} | "
        f"calibration={row['calibration_holes']} | "
        f"monitoring={row['monitoring_holes']}"
    )

df_eligibility.to_csv(
    os.path.join(RESULTS_DIR, "drill_eligibility_ultrasonic.csv"),
    index=False,
)

for drill_name in sorted(drills_data):
    print(f"  {drill_name}: {len(drills_data[drill_name]['X'])} holes")

comparison_records = []
continuous_mlp_data = []
continuous_lstm_data = []
continuous_xgb_data = []
threshold_records = []
metric_records = []

FINE_TUNING_VARIANTS = [
    "FT_LAST_LAYER",
    "FT_DECODER",
    "FT_ALL_LOW_LR",
]
finetuning_metric_records = []
finetuning_score_records = []


def diagnostic_persistence_scores(
        scores,
        threshold,
        y_true,
        start_index=EVAL_START_HOLE_INDEX,
        window=PERSISTENCE_W,
):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    raw_alarm = np.zeros(len(scores), dtype=int)

    valid = np.isfinite(scores)
    raw_alarm[valid] = (
            scores[valid] > threshold
    ).astype(int)

    valid_scores = scores[start_index:]
    valid_true = y_true[start_index:]
    valid_raw = raw_alarm[start_index:]

    max_run = 0
    current_run = 0

    for value in valid_raw:
        if value == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    raw_crossings = np.where(valid_raw == 1)[0]

    first_raw = (
        int(raw_crossings[0] + start_index + 1)
        if len(raw_crossings) > 0
        else None
    )

    first_persistent = None

    if len(valid_raw) >= window:
        for i in range(window - 1, len(valid_raw)):
            current = valid_raw[
                i - window + 1 : i + 1
            ]

            if np.all(current == 1):
                first_persistent = int(
                    i + start_index + 1
                )
                break

    failure_mask = valid_true == 1
    normal_mask = valid_true == 0

    print("  [DIAGNOSIS OF PERSISTENCE]")
    print(f"    threshold:              {threshold:.6f}")
    print(f"    raw alarms:             {int(valid_raw.sum())}")
    print(f"    max consecutive alarms: {max_run}")
    print(f"    first raw crossing:     {first_raw}")
    print(f"    first persistent:       {first_persistent}")

    if np.any(normal_mask):
        print(
            f"    normal max score:       "
            f"{np.nanmax(valid_scores[normal_mask]):.6f}"
        )

    if np.any(failure_mask):
        print(
            f"    failure max score:      "
            f"{np.nanmax(valid_scores[failure_mask]):.6f}"
        )

        print(
            f"    failure mean score:     "
            f"{np.nanmean(valid_scores[failure_mask]):.6f}"
        )

    print(
        f"    scores > threshold:     "
        f"{int(valid_raw.sum())}/{len(valid_raw)}"
    )
for drill_test in eligible_drills:
    print(f"LODO — TEST DRILL: {drill_test}")

    K.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_test = drills_data[drill_test]["X"]
    y_test = drills_data[drill_test]["y"]
    test_holes = drills_data[drill_test]["holes"]
    n_holes = len(X_test)

    if n_holes <= CALIBRATION_HOLES:
        continue

    X_train_xgb, y_train_xgb = build_training_data(drills_data, drill_test)
    X_normal_train, normal_by_drill = build_ae_normal_training_data(
        drills_data, drill_test
    )

    print(f"XGBoost training: {len(X_train_xgb)} events")
    print(f"Normal AE training: {len(X_normal_train)} events")

    scaler_xgb = StandardScaler()
    X_train_xgb_sc = scaler_xgb.fit_transform(X_train_xgb)
    X_test_xgb_sc = scaler_xgb.transform(X_test)

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=SEED,
        objective="binary:logistic",
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    )
    xgb.fit(X_train_xgb_sc, y_train_xgb)

    xgb_probs = xgb.predict_proba(X_test_xgb_sc)[:, 1]
    raw_alarm_xgb = (xgb_probs >= XGB_THRESHOLD).astype(int)

    scaler_mlp, mlp_ae = train_mlp_autoencoder(
        X_normal_train, X_normal_train.shape[1]
    )

    threshold_mlp, baseline_train_mlp, train_scores_mlp = (
        compute_global_threshold_from_training_scores(
            mlp_ae,
            scaler_mlp,
            normal_by_drill,
            model_type="mlp",
        )
    )

    X_test_mlp_sc = scaler_mlp.transform(X_test)
    mlp_test_errors = reconstruction_errors_mlp(mlp_ae, X_test_mlp_sc)

    mlp_scores_test, baseline_mlp_test = normalize_by_initial_baseline(
        mlp_test_errors
    )
    raw_alarm_mlp = (
        mlp_scores_test > threshold_mlp
    ).astype(int)
    raw_alarm_mlp[:CALIBRATION_HOLES] = 0

    calibration_errors_mlp = mlp_test_errors[:CALIBRATION_HOLES]
    calibration_median_mlp = baseline_mlp_test

    monitoring_scores_mlp = mlp_scores_test[CALIBRATION_HOLES:]

    normal_monitoring_mlp = monitoring_scores_mlp[
        y_test[CALIBRATION_HOLES:] == 0
    ]

    failure_monitoring_mlp = monitoring_scores_mlp[
        y_test[CALIBRATION_HOLES:] == 1
    ]

    print("\nMLP")
    print(f"  global threshold:       {threshold_mlp:.6f}")
    print(f"  local baseline:         {calibration_median_mlp:.6f}")
    print(
        f"  calibration errors:     "
        f"{np.round(calibration_errors_mlp, 6).tolist()}"
    )
    print(
        f"  calibration normalized: "
        f"{np.round(mlp_scores_test[:CALIBRATION_HOLES], 6).tolist()}"
    )

    if len(monitoring_scores_mlp) > 0:
        print(
            f"  monitoring score min:   "
            f"{np.nanmin(monitoring_scores_mlp):.6f}"
        )
        print(
            f"  monitoring score max:   "
            f"{np.nanmax(monitoring_scores_mlp):.6f}"
        )

    if len(normal_monitoring_mlp) > 0:
        print(
            f"  normal score max:       "
            f"{np.nanmax(normal_monitoring_mlp):.6f}"
        )
    else:
        print("  normal score max:       N/A")

    if len(failure_monitoring_mlp) > 0:
        print(
            f"  failure score max:      "
            f"{np.nanmax(failure_monitoring_mlp):.6f}"
        )
    else:
        print("  failure score max:      N/A")
    diagnostic_persistence_scores(
        mlp_scores_test,
        threshold_mlp,
        y_test,
    )

    scaler_lstm = StandardScaler()
    scaler_lstm.fit(X_normal_train)

    lstm_ae, X_train_seq_lstm = train_lstm_autoencoder(
        normal_by_drill,
        scaler_lstm,
    )

    threshold_lstm, baseline_train_lstm, train_scores_lstm = (
        compute_global_threshold_from_training_scores(
            lstm_ae,
            scaler_lstm,
            normal_by_drill,
            model_type="lstm",
        )
    )

    X_test_lstm_sc = scaler_lstm.transform(X_test)

    X_test_seq_lstm = create_padded_causal_sequences(
        X_test_lstm_sc,
        SEQ_LEN,
    )

    errors_seq_lstm = reconstruction_errors_lstm(
        lstm_ae,
        X_test_seq_lstm,
    )

    lstm_test_errors = map_lstm_sequence_errors_to_holes(
        errors_seq_lstm,
        n_holes,
    )

    lstm_scores_test, baseline_lstm_test = normalize_by_initial_baseline(
        lstm_test_errors
    )
    raw_alarm_lstm = np.zeros(n_holes, dtype=int)
    valid_lstm = np.isfinite(lstm_scores_test)
    raw_alarm_lstm[valid_lstm] = (
        lstm_scores_test[valid_lstm] > threshold_lstm
    ).astype(int)
    raw_alarm_lstm[:CALIBRATION_HOLES] = 0

    calibration_errors_lstm = lstm_test_errors[:CALIBRATION_HOLES]
    calibration_median_lstm = baseline_lstm_test

    monitoring_scores_lstm = lstm_scores_test[CALIBRATION_HOLES:]

    normal_monitoring_lstm = monitoring_scores_lstm[
        y_test[CALIBRATION_HOLES:] == 0
    ]

    failure_monitoring_lstm = monitoring_scores_lstm[
        y_test[CALIBRATION_HOLES:] == 1
    ]

    print("\nLSTM")
    print(f"  global threshold:       {threshold_lstm:.6f}")
    print(f"  local baseline:         {calibration_median_lstm:.6f}")
    print(
        f"  calibration errors:     "
        f"{np.round(calibration_errors_lstm, 6).tolist()}"
    )
    print(
        f"  calibration normalized: "
        f"{np.round(lstm_scores_test[:CALIBRATION_HOLES], 6).tolist()}"
    )

    if len(monitoring_scores_lstm) > 0:
        print(
            f"  monitoring score min:   "
            f"{np.nanmin(monitoring_scores_lstm):.6f}"
        )
        print(
            f"  monitoring score max:   "
            f"{np.nanmax(monitoring_scores_lstm):.6f}"
        )

    if len(normal_monitoring_lstm) > 0:
        print(
            f"  normal score max:       "
            f"{np.nanmax(normal_monitoring_lstm):.6f}"
        )
    else:
        print("  normal score max:       N/A")

    if len(failure_monitoring_lstm) > 0:
        print(
            f"  failure score max:      "
            f"{np.nanmax(failure_monitoring_lstm):.6f}"
        )
    else:
        print("  failure score max:      N/A")
    diagnostic_persistence_scores(
        lstm_scores_test,
        threshold_lstm,
        y_test,
    )

    X_calibration_mlp_sc = X_test_mlp_sc[:CALIBRATION_HOLES]
    X_calibration_lstm_sc = X_test_lstm_sc[:CALIBRATION_HOLES]

    for variant in FINE_TUNING_VARIANTS:
        print("\n" + "-" * 60)
        print(f"MLP-AE fine-tuning: {variant}")
        print("-" * 60)

        ft_mlp_model = fine_tune_mlp_on_calibration(
            mlp_ae,
            X_calibration_mlp_sc,
            variant,
        )

        ft_mlp_errors = reconstruction_errors_mlp(
            ft_mlp_model,
            X_test_mlp_sc,
        )
        ft_mlp_scores, ft_mlp_local_baseline = normalize_by_initial_baseline(
            ft_mlp_errors
        )

        ft_mlp_raw_alarm = np.zeros(n_holes, dtype=int)
        valid_ft_mlp = np.isfinite(ft_mlp_scores)
        ft_mlp_raw_alarm[valid_ft_mlp] = (
            ft_mlp_scores[valid_ft_mlp] > threshold_mlp
        ).astype(int)
        ft_mlp_raw_alarm[:CALIBRATION_HOLES] = 0

        ft_mlp_prediction = persistence_alarm(ft_mlp_raw_alarm)
        ft_mlp_metrics = evaluate_binary(ft_mlp_prediction, y_test)

        finetuning_metric_records.append(
            {
                "drill": drill_test,
                "model": "MLP-AE",
                "variant": variant,
                "accuracy": ft_mlp_metrics["accuracy"],
                "precision": ft_mlp_metrics["precision"],
                "recall": ft_mlp_metrics["recall"],
                "f1": ft_mlp_metrics["f1"],
                "n": ft_mlp_metrics["n"],
                "global_threshold": threshold_mlp,
                "local_baseline": ft_mlp_local_baseline,
                "fine_tuning_holes": CALIBRATION_HOLES,
                "fine_tuning_epochs": 10,
                "fine_tuning_learning_rate": (
                    1e-3 if variant != "FT_ALL_LOW_LR" else 1e-4
                ),
            }
        )

        for i in range(n_holes):
            finetuning_score_records.append(
                {
                    "drill": drill_test,
                    "model": "MLP-AE",
                    "variant": variant,
                    "hole": test_holes[i],
                    "hole_index": i + 1,
                    "phase": "calibration" if i < CALIBRATION_HOLES else "monitoring",
                    "mse": float(ft_mlp_errors[i]) if np.isfinite(ft_mlp_errors[i]) else np.nan,
                    "normalized_score": float(ft_mlp_scores[i]) if np.isfinite(ft_mlp_scores[i]) else np.nan,
                    "threshold": float(threshold_mlp),
                    "raw_alarm": int(ft_mlp_raw_alarm[i]) if i >= CALIBRATION_HOLES else 0,
                    "prediction": int(ft_mlp_prediction[i]) if i >= CALIBRATION_HOLES else 0,
                    "label": int(y_test[i]),
                }
            )

        print(
            f"{drill_test} | MLP-AE | {variant} | "
            f"F1={ft_mlp_metrics['f1']:.3f} | "
            f"Precision={ft_mlp_metrics['precision']:.3f} | "
            f"Recall={ft_mlp_metrics['recall']:.3f}"
        )

        print(f"LSTM-AE fine-tuning: {variant}")

        ft_lstm_model = fine_tune_lstm_on_calibration(
            lstm_ae,
            X_calibration_lstm_sc,
            variant,
        )

        ft_lstm_seq = create_padded_causal_sequences(
            X_test_lstm_sc,
            SEQ_LEN,
        )
        ft_lstm_errors = reconstruction_errors_lstm(ft_lstm_model, ft_lstm_seq)
        ft_lstm_errors = map_lstm_sequence_errors_to_holes(
            ft_lstm_errors,
            n_holes,
        )

        ft_lstm_scores, ft_lstm_local_baseline = normalize_by_initial_baseline(
            ft_lstm_errors
        )

        ft_lstm_raw_alarm = np.zeros(n_holes, dtype=int)
        valid_ft_lstm = np.isfinite(ft_lstm_scores)
        ft_lstm_raw_alarm[valid_ft_lstm] = (
            ft_lstm_scores[valid_ft_lstm] > threshold_lstm
        ).astype(int)
        ft_lstm_raw_alarm[:CALIBRATION_HOLES] = 0

        ft_lstm_prediction = persistence_alarm(ft_lstm_raw_alarm)
        ft_lstm_metrics = evaluate_binary(ft_lstm_prediction, y_test)

        finetuning_metric_records.append(
            {
                "drill": drill_test,
                "model": "LSTM-AE",
                "variant": variant,
                "accuracy": ft_lstm_metrics["accuracy"],
                "precision": ft_lstm_metrics["precision"],
                "recall": ft_lstm_metrics["recall"],
                "f1": ft_lstm_metrics["f1"],
                "n": ft_lstm_metrics["n"],
                "global_threshold": threshold_lstm,
                "local_baseline": ft_lstm_local_baseline,
                "fine_tuning_holes": CALIBRATION_HOLES,
                "fine_tuning_epochs": 10,
                "fine_tuning_learning_rate": (
                    1e-3 if variant != "FT_ALL_LOW_LR" else 1e-4
                ),
            }
        )

        for i in range(n_holes):
            finetuning_score_records.append(
                {
                    "drill": drill_test,
                    "model": "LSTM-AE",
                    "variant": variant,
                    "hole": test_holes[i],
                    "hole_index": i + 1,
                    "phase": "calibration" if i < CALIBRATION_HOLES else "monitoring",
                    "mse": float(ft_lstm_errors[i]) if np.isfinite(ft_lstm_errors[i]) else np.nan,
                    "normalized_score": float(ft_lstm_scores[i]) if np.isfinite(ft_lstm_scores[i]) else np.nan,
                    "threshold": float(threshold_lstm),
                    "raw_alarm": int(ft_lstm_raw_alarm[i]) if i >= CALIBRATION_HOLES else 0,
                    "prediction": int(ft_lstm_prediction[i]) if i >= CALIBRATION_HOLES else 0,
                    "label": int(y_test[i]),
                }
            )

        print(
            f"{drill_test} | LSTM-AE | {variant} | "
            f"F1={ft_lstm_metrics['f1']:.3f} | "
            f"Precision={ft_lstm_metrics['precision']:.3f} | "
            f"Recall={ft_lstm_metrics['recall']:.3f}"
        )

    pred_xgb = persistence_alarm(raw_alarm_xgb)
    pred_mlp = persistence_alarm(raw_alarm_mlp)
    pred_lstm = persistence_alarm(raw_alarm_lstm)

    threshold_records.append(
        {
            "drill_test": drill_test,
            "threshold_mlp": threshold_mlp,
            "threshold_lstm": threshold_lstm,
            "baseline_mlp_test": baseline_mlp_test,
            "baseline_lstm_test": baseline_lstm_test,
            "calibration_holes": CALIBRATION_HOLES,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "n_training_drills_total": len(drills_data) - 1,
            "n_training_drills_ae_normal": len(normal_by_drill),
            "n_threshold_scores_mlp": len(train_scores_mlp),
            "n_threshold_scores_lstm": len(train_scores_lstm),
        }
    )

    for i in range(n_holes):
        hole = test_holes[i]
        is_calibration = i < CALIBRATION_HOLES
        is_monitoring = not is_calibration

        comparison_records.append(
            {
                "drill": drill_test,
                "hole": hole,
                "hole_index": i + 1,
                "phase": "calibration" if is_calibration else "monitoring",
                "alarm_xgb": int(pred_xgb[i]) if is_monitoring else 0,
                "alarm_mlp": int(pred_mlp[i]) if is_monitoring else 0,
                "alarm_lstm": int(pred_lstm[i]) if is_monitoring else 0,
            }
        )

        continuous_mlp_data.append(
            {
                "drill": drill_test,
                "hole": hole,
                "hole_index": i + 1,
                "phase": "calibration" if is_calibration else "monitoring",
                "mse": float(mlp_test_errors[i]),
                "normalized_score": float(mlp_scores_test[i]),
                "threshold": float(threshold_mlp),
                "prediction": int(pred_mlp[i]) if is_monitoring else 0,
                "raw_alarm": int(raw_alarm_mlp[i]) if is_monitoring else 0,
                "label": int(y_test[i]),
            }
        )

        continuous_lstm_data.append(
            {
                "drill": drill_test,
                "hole": hole,
                "hole_index": i + 1,
                "phase": "calibration" if is_calibration else "monitoring",
                "mse": (
                    float(lstm_test_errors[i])
                    if np.isfinite(lstm_test_errors[i])
                    else np.nan
                ),
                "normalized_score": (
                    float(lstm_scores_test[i])
                    if np.isfinite(lstm_scores_test[i])
                    else np.nan
                ),
                "threshold": float(threshold_lstm),
                "prediction": int(pred_lstm[i]) if is_monitoring else 0,
                "raw_alarm": int(raw_alarm_lstm[i]) if is_monitoring else 0,
                "label": int(y_test[i]),
            }
        )

        continuous_xgb_data.append(
            {
                "drill": drill_test,
                "hole": hole,
                "hole_index": i + 1,
                "phase": "calibration" if is_calibration else "monitoring",
                "score": float(xgb_probs[i]),
                "threshold": XGB_THRESHOLD,
                "prediction": int(pred_xgb[i]) if is_monitoring else 0,
                "raw_alarm": int(raw_alarm_xgb[i]) if is_monitoring else 0,
                "label": int(y_test[i]),
            }
        )

    mlp_metrics = evaluate_binary(pred_mlp, y_test)
    lstm_metrics = evaluate_binary(pred_lstm, y_test)
    xgb_metrics = evaluate_binary(pred_xgb, y_test)

    metric_records.extend(
        [
            {"drill": drill_test, "model": "MLP-AE", **mlp_metrics},
            {"drill": drill_test, "model": "LSTM-AE", **lstm_metrics},
            {"drill": drill_test, "model": "XGBoost", **xgb_metrics},
        ]
    )

    print(
        f"{drill_test} | MLP F1={mlp_metrics['f1']:.3f} | "
        f"LSTM F1={lstm_metrics['f1']:.3f} | "
        f"XGB F1={xgb_metrics['f1']:.3f}"
    )

df_models = pd.DataFrame(comparison_records)
df_mlp = pd.DataFrame(continuous_mlp_data)
df_lstm = pd.DataFrame(continuous_lstm_data)
df_xgb = pd.DataFrame(continuous_xgb_data)
df_thresholds = pd.DataFrame(threshold_records)
df_metrics = pd.DataFrame(metric_records)
df_finetuning_metrics = pd.DataFrame(finetuning_metric_records)
df_finetuning_scores = pd.DataFrame(finetuning_score_records)

if df_models.empty:
    raise ValueError(
        "No drill produced at least 6 holes for the calibration + monitoring protocol."
    )

save_roc_data(
    df_mlp,
    "normalized_score",
    os.path.join(RESULTS_DIR, "ultrasonic_mlp_roc_data.csv"),
)

save_roc_data(
    df_lstm,
    "normalized_score",
    os.path.join(RESULTS_DIR, "ultrasonic_lstm_roc_data.csv"),
)

save_roc_data(
    df_xgb,
    "score",
    os.path.join(RESULTS_DIR, "ultrasonic_xgboost_roc_data.csv"),
)
for model_name in ["MLP-AE", "LSTM-AE"]:
    model_tag = "mlp" if model_name == "MLP-AE" else "lstm"
    for variant in FINE_TUNING_VARIANTS:
        df_variant = df_finetuning_scores[
            (df_finetuning_scores["model"] == model_name)
            & (df_finetuning_scores["variant"] == variant)
        ]

        if not df_variant.empty:
            save_roc_data(
                df_variant,
                "normalized_score",
                os.path.join(
                    RESULTS_DIR,
                    f"ultrasonic_{model_tag}_{variant.lower()}_roc_data.csv",
                ),
            )
df_final = pd.merge(
    df_models,
    df_listening,
    on=["drill", "hole"],
    how="inner",
)

print(f"Total of holes in the models:      {len(df_models)}")
print(f"Total of holes in the listening:   {len(df_listening)}")
print(f"Total after merge:                 {len(df_final)}")
print(
    f"Drills evaluated: {df_models['drill'].nunique()} / "
    f"{len(drills_data)}"
)

if df_final.empty:
    raise ValueError(
        "CRITICAL ERROR: The merge between the models and the manual review data resulted in an empty dataframe."
    )

df_mlp.to_csv(
    os.path.join(RESULTS_DIR, "results_autoencoder_ultrasonic.csv"),
    index=False,
)
df_lstm.to_csv(
    os.path.join(RESULTS_DIR, "results_LSTM_ultrasonic.csv"),
    index=False,
)
df_xgb.to_csv(
    os.path.join(RESULTS_DIR, "results_xgboost_ultrasonic.csv"),
    index=False,
)
df_models.to_csv(
    os.path.join(RESULTS_DIR, "results_comparison_ultrasonic.csv"),
    index=False,
)
df_thresholds.to_csv(
    os.path.join(RESULTS_DIR, "thresholds_ultrasonic.csv"),
    index=False,
)
df_metrics.to_csv(
    os.path.join(RESULTS_DIR, "metrics_per_drill_ultrasonic.csv"),
    index=False,
)

df_finetuning_metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "metrics_per_drill_mlp_lstm_finetuning_3_variants_ultrasonic.csv",
    ),
    index=False,
)
df_finetuning_scores.to_csv(
    os.path.join(
        RESULTS_DIR,
        "results_MLP_LSTM_finetuning_3_variants_ultrasonic.csv",
    ),
    index=False,
)

if "results" in globals():
    try:
        pd.DataFrame(RESULTS_DIR).to_csv(
            "fine_tuning_per_drill_results_ultrasonic.csv",
            index=False
        )
        print("\nSaved: fine_tuning_per_drill_results_ultrasonic.csv")
    except Exception as exc:
        print(f"\nCould not save results CSV: {exc}")