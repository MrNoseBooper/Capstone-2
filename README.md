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
