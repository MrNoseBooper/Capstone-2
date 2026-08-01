"""
Turns the raw consolidated PhysioNet dataframe into:
  1. A TABULAR feature set (current vitals + rolling-window summary stats +
     demographics + missingness flags) -> used by Random Forest, XGBoost, and KAN.
  2. A SEQUENTIAL feature set (fixed-length windows of per-hour vitals) ->
     used by the LSTM.

Splitting is done at the PATIENT level (never by row) so that no patient's
hours leak between train / validation / test.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

FEATURE_COLS = config.VITAL_COLUMNS + config.STATIC_COLUMNS


def _add_missingness_flags(df, cols):
    for c in cols:
        df[f"{c}_missing"] = df[c].isnull().astype(int)
    return df


def _forward_fill_within_patient(df, cols):
    df = df.sort_values(["PatientID", "ICULOS"])
    df[cols] = df.groupby("PatientID")[cols].ffill()
    return df


def clean_and_engineer(df):
    """Applies missingness flags, forward-fill, and rolling-window tabular features."""
    df = df.copy()
    df = _add_missingness_flags(df, config.VITAL_COLUMNS)
    df = _forward_fill_within_patient(df, config.VITAL_COLUMNS)

    # Rolling-window summary statistics per patient (mean/std/min/max over
    # the past WINDOW_SIZE hours) -- these become the tree-model features.
    df = df.sort_values(["PatientID", "ICULOS"])
    grouped = df.groupby("PatientID")[config.VITAL_COLUMNS]
    roll = grouped.rolling(window=config.WINDOW_SIZE, min_periods=1)

    means = roll.mean().reset_index(level=0, drop=True).add_suffix("_rollmean")
    stds = roll.std().reset_index(level=0, drop=True).add_suffix("_rollstd")
    mins = roll.min().reset_index(level=0, drop=True).add_suffix("_rollmin")
    maxs = roll.max().reset_index(level=0, drop=True).add_suffix("_rollmax")

    df = pd.concat([df, means, stds, mins, maxs], axis=1)
    return df


def impute_remaining(train_df, other_dfs, cols):
    """Fill any still-missing values using TRAIN-set medians only (no leakage)."""
    medians = train_df[cols].median()
    train_df[cols] = train_df[cols].fillna(medians)
    filled_others = []
    for d in other_dfs:
        d = d.copy()
        d[cols] = d[cols].fillna(medians)
        filled_others.append(d)
    return train_df, filled_others, medians


def patient_level_split(df, test_size=None, val_size=None, seed=None):
    """Splits patients (not rows) into train / val / test."""
    test_size = test_size if test_size is not None else config.TEST_SIZE
    val_size = val_size if val_size is not None else config.VAL_SIZE
    seed = seed if seed is not None else config.RANDOM_SEED

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["PatientID"]))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["PatientID"]))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    print(f"[split] patients -> train={train_df['PatientID'].nunique()}, "
          f"val={val_df['PatientID'].nunique()}, test={test_df['PatientID'].nunique()}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_tabular_feature_columns(df):
    engineered_suffixes = ("_rollmean", "_rollstd", "_rollmin", "_rollmax", "_missing")
    cols = [c for c in df.columns if c in FEATURE_COLS or c.endswith(engineered_suffixes)]
    return cols


def build_sequences(df, window_size=None):
    """
    Builds fixed-length sequences (window_size x n_vitals) per patient-hour
    for the LSTM. Early hours are left-padded by repeating the first
    available reading. Returns X (n_samples, window, n_features), y, and
    the row index used (to align with tabular features / IDs if needed).
    """
    window_size = window_size or config.WINDOW_SIZE
    cols = config.VITAL_COLUMNS
    sequences = []
    labels = []

    for pid, g in df.groupby("PatientID"):
        g = g.sort_values("ICULOS")
        values = g[cols].to_numpy(dtype=np.float32)
        y = g["SepsisLabel"].to_numpy()

        n = len(g)
        for i in range(n):
            start = max(0, i - window_size + 1)
            window = values[start:i + 1]
            pad_len = window_size - window.shape[0]
            if pad_len > 0:
                pad = np.repeat(window[0:1], pad_len, axis=0) if window.shape[0] > 0 else np.zeros((pad_len, len(cols)), dtype=np.float32)
                window = np.vstack([pad, window])
            sequences.append(window)
            labels.append(y[i])

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y
