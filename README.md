# Acoustic Monitoring for Drill Jamming Detection Under Inter-Tool Variability

This repository accompanies **Acoustic Monitoring for Drill Jamming Detection Under Inter-Tool Variability**, prepared for SAFEPROCESS 2027. It documents the evaluation of acoustic drill-jamming monitoring under inter-tool variability using synchronized audible and ultrasonic recordings from 20 carbide drills.

The study extends the earlier ultrasonic investigation, *Acoustic Anomaly Detection for Early Identification of Drill Jamming* (INDIN 2026), by considering audible and hybrid sensing, a temporal LSTM autoencoder, local calibration and fine-tuning, persistent alarms, and complementary evaluation analyses.

## Experimental overview

The experiments monitored 4 mm carbide drills while drilling iron plates on a CNC machining center until an observed jamming event. Sequence lengths differed between drills. Audible signals were acquired at **96 kHz** and ultrasonic signals at **192 kHz**, synchronously from the same internal sensor position (**Channel 4**).

Three models are evaluated:

- **XGBoost:** supervised classification of retrospectively defined operational regions.
- **MLP-AE:** semi-supervised reconstruction-based detection using one feature vector per drilling event.
- **LSTM-AE:** semi-supervised temporal reconstruction using sequences of five consecutive drilling events.

Each model is evaluated with **audible**, **ultrasonic**, and **hybrid** feature configurations. Hybrid sensing concatenates the feature vectors from both acoustic domains; it does not perform spatial localization.

## Retrospective operational labels

Independent measurements of physical tool degradation were not available. Labels are assigned retrospectively according to each event's relative position in the observed sequence:

| Portion of each drill sequence | Operational region | Label |
| --- | --- | ---: |
| First 50% | Normal | `0` |
| Following 30% | Transition | `-1` |
| Final 20% | Failure-aligned | `1` |

These regions are **operational evaluation references, not independently measured wear stages**. Transition events are excluded from binary evaluation.

## Signal processing and features

Recordings are segmented by hole, with each drilling event treated as one observation. Frame-level descriptors use a 1024-sample window and 512-sample hop (50% overlap) and are aggregated into one vector per event.

For each acoustic domain, the mean and standard deviation of 20 MFCC coefficients, plus the mean and standard deviation of RMS energy, yield **42 features**. The hybrid configuration concatenates both vectors for **84 features**.

## Validation and calibration

Evaluation uses **Leave-One-Drill-Out (LODO)** validation: one drill is held out in each of 20 folds, and the remaining drills supply training observations. The first **five physical holes** of the held-out drill are reserved for local calibration; monitoring starts at **hole 6**. No future labels from the held-out drill are used for calibration.

The autoencoders learn from normal training events. Their anomaly thresholds are derived from the **99.5th percentile** of reconstruction errors normalized using drill-specific training baselines. On the held-out drill, reconstruction-error scores are locally normalized using the median error of its first five holes. Thresholds are fixed during testing and across adaptation variants.

The following unlabeled autoencoder adaptation strategies are compared independently against the baseline without fine-tuning:

1. Full-network fine-tuning with a reduced learning rate.
2. Decoder-only fine-tuning.
3. Last-layer fine-tuning.

XGBoost uses labeled normal and failure-aligned events from the training drills; transition events are excluded from its training. Its binary decision threshold is 0.5.

## Alarm and evaluation protocol

A **definitive alarm** requires two consecutive monitored events satisfying the model's anomaly criterion. This persistence rule applies to all three models.

- **Event-level:** Precision, Recall, Accuracy, and F1-score are computed from final predictions after alarm persistence; AUC uses continuous model scores. Only monitored normal and failure-aligned events are included, excluding calibration and transition events.
- **Regional:** Each drill contributes one decision for its monitored normal region and one for its failure-aligned region. A region is alert-positive if at least one definitive alarm occurs within it (40 regional decisions per model across 20 drills).
- **Temporal:** Mean Lead Time (MLT) counts holes between the first eligible definitive alarm and the beginning of the failure-aligned region. Only alarms confirmed after calibration and before that boundary qualify; drills without one receive zero. MLT is not remaining useful life or physical time to jamming.
- **Statistical:** Paired per-drill F1-scores for hybrid sensing are compared with two-sided Wilcoxon signed-rank tests and Holm correction for three pairwise comparisons.

## Repository layout

The layout below describes the intended organization. Include only directories and scripts that actually exist in the published revision, and keep input paths consistent with the scripts.

```text
.
├── data/
│   ├── raw/                 # Original recordings, if distributed locally
│   ├── segmented/           # Hole-level recordings, if distributed locally
│   ├── metadata/            # Available experimental metadata
│   ├── processed/           # Extracted features, if distributed locally
│   └── results/             # Model prediction CSVs and generated summaries
├── plots/                   # Exported PDF and PNG figures
├── src/
│   ├── preprocessing/       # Segmentation and feature extraction
│   ├── models/              # Training, calibration, and inference
│   ├── analysis/            # Metrics and statistical analysis
│   └── visualization/       # Figure-generation scripts
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```