# Sepsis Early-Prediction Model Comparison

Compares **4 models** — Random Forest, XGBoost (machine learning) and
LSTM, KAN (deep learning) — on the task of predicting sepsis onset
6 hours before it clinically occurs, using the PhysioNet/Computing in
Cardiology Challenge 2019 dataset (real ICU vitals from 40,336 patients
across two hospital systems).

## 1. Setup

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, xgboost, torch, matplotlib.
(If you don't have a GPU, torch will run on CPU automatically — it'll
just be slower for the LSTM/KAN steps. See the speed tips below.)

## 2. Add the data

Extract the PhysioNet Challenge 2019 archive (the one with
`training_setA` / `training_setB` folders full of `.psv` files) so that
it ends up somewhere inside `data/raw/`. For example:

```
data/raw/training_setA/training/p00001.psv ...
data/raw/training_setB/training_setB/p100001.psv ...
```

The exact nesting doesn't matter — `data_loader.py` searches recursively
for every `.psv` file under `data/raw/`.

## 3. Run everything

```bash
python -m src.main
```

This will:
1. Consolidate all `.psv` files into one dataframe (cached afterwards to
   `data/processed_sepsis.csv` so you don't have to re-parse next time —
   delete that file, or pass `--force-reload`, if you change the raw data).
2. Engineer features: missingness flags, per-patient forward-fill, and
   rolling-window statistics (mean/std/min/max over the past 8 hours).
3. Split **by patient** (not by row) into train/val/test so no patient's
   hours leak across the split.
4. Train all 4 models and evaluate them on the same held-out test
   patients.
5. Save a comparison table and plots to `outputs/`.

### Useful flags

```bash
# Quick test run on a small subset of patients (much faster)
python -m src.main --max-patients 2000

# Skip the slower deep learning models while you're debugging
python -m src.main --max-patients 2000 --skip-lstm --skip-kan

# Re-parse raw .psv files even if a cached CSV already exists
python -m src.main --force-reload
```

All hyperparameters (window size, model sizes, epochs, train/test split
ratios, etc.) live in `config.py` — edit them there rather than in the
scripts.

## 4. Outputs

After a run, `outputs/` will contain:
- `model_comparison_results.csv` — AUROC, AUPRC, F1, Precision, Recall,
  Accuracy for all 4 models, sorted best-to-worst by AUROC
- `metrics_comparison_bar.png` — bar chart comparing all metrics across models
- `roc_curves_comparison.png` — overlaid ROC curves for all 4 models
- `confusion_matrix_<model>.png` — one per model

## 5. Design notes (for your report)

- **Why AUROC/AUPRC over plain accuracy:** sepsis is rare (~1.8% of
  hourly readings, ~7.3% of patients ever septic), so a model that
  always predicts "no sepsis" would score >98% accuracy while being
  useless. AUPRC in particular is more informative under this kind of
  class imbalance.
- **Why patient-level splitting matters:** splitting by row would let
  hours from the same ICU stay appear in both train and test, which
  leaks information and inflates reported performance. This pipeline
  always splits by `PatientID`.
- **Why RF/XGBoost/KAN get different features than LSTM:** tree-based
  models and KAN work on tabular feature vectors (current vitals +
  rolling-window summary statistics + demographics), which is standard
  practice for these architectures. The LSTM instead receives the raw
  temporal sequence (an 8-hour window of vitals per prediction point) so
  it can learn temporal dynamics directly — this is the reason LSTMs are
  used for this kind of problem in the literature. This asymmetry is a
  deliberate, defensible methodological choice, not an inconsistency —
  worth stating explicitly in your methodology chapter.
- **Class imbalance handling:** Random Forest uses `class_weight="balanced"`;
  XGBoost uses `scale_pos_weight`; the LSTM/KAN loss functions use a
  weighted `BCEWithLogitsLoss`. All 4 models see the same effective
  imbalance-correction philosophy, keeping the comparison fair.
- **Missing data:** real ICU vitals/labs have heavy missingness (see the
  audit you already have). This pipeline forward-fills within each
  patient (carrying the last known reading forward, which mirrors how a
  real monitor would behave between measurements) and imputes any
  remaining gaps using **training-set-only** medians, to avoid leaking
  test-set statistics into training.

## 6. Project structure

```
sepsis_model_comparison/
├── config.py                  # all paths & hyperparameters
├── requirements.txt
├── data/
│   └── raw/                   # <- put the extracted PhysioNet data here
├── outputs/                   # <- results land here after a run
└── src/
    ├── data_loader.py         # consolidates .psv files -> one dataframe
    ├── feature_engineering.py # missingness flags, rolling stats, patient-level split
    ├── evaluate.py            # metrics + comparison plots
    ├── main.py                # orchestrates the full pipeline
    └── models/
        ├── tree_models.py     # Random Forest, XGBoost
        ├── deep_models.py     # LSTM, KAN training loops (PyTorch)
        └── kan_layer.py       # from-scratch KAN spline layer (no extra KAN library needed)
```

## 7. Speed tips

The full dataset is ~1.55M patient-hours. On CPU, LSTM/KAN training over
this many rows can take a while. If you just want to confirm everything
runs and get a first set of comparison numbers, start with:

```bash
python -m src.main --max-patients 5000
```

then re-run without `--max-patients` (or with a larger number) once
you're happy with the pipeline, for your final report numbers.
