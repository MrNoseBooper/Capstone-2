"""
Central configuration for the sepsis early-prediction model comparison project.
Edit the paths / hyperparameters here rather than inside the scripts.
"""
import os

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
# Point this at the folder that contains the extracted PhysioNet/CinC
# 2019 Challenge data. It should contain (somewhere inside it, any depth)
# the training_setA and training_setB folders full of .psv files.
RAW_DATA_DIR = os.path.join("data", "raw")

# Where the consolidated, cleaned dataset is cached after the first run
# so you don't have to re-parse 40k+ .psv files every time.
PROCESSED_DATA_PATH = os.path.join("data", "processed_sepsis.csv")

# Where all results (metrics table, plots, trained model files) are written
OUTPUT_DIR = "outputs"

# ------------------------------------------------------------------
# DATA / SAMPLING
# ------------------------------------------------------------------
# Set to an integer (e.g. 5000) to subsample patients for a quick test run.
# Set to None to use the full 40,336-patient dataset (slower, especially LSTM/KAN).
MAX_PATIENTS = None

RANDOM_SEED = 42

# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------
# Core vital-sign columns used for prediction (chosen for relatively low
# missingness so the model is trained on genuinely available IoT-style signals)
VITAL_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

# Static / demographic columns
STATIC_COLUMNS = ["Age", "Gender", "HospAdmTime"]

# Sliding window length (in hours) used for temporal feature construction
WINDOW_SIZE = 8

# Test set proportion (split by PATIENT, not by row, to avoid leakage)
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # taken out of the remaining training patients

# ------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

LSTM_PARAMS = dict(
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    epochs=15,
    batch_size=256,
    lr=1e-3,
)

KAN_PARAMS = dict(
    hidden_sizes=[64, 32],
    grid_size=5,
    spline_order=3,
    epochs=15,
    batch_size=256,
    lr=1e-3,
)
