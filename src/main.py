"""
End-to-end pipeline:
  1. Load / consolidate raw PhysioNet sepsis data
  2. Clean + engineer features (missingness flags, forward-fill, rolling stats)
  3. Split by PATIENT into train / val / test
  4. Build tabular feature matrices (RF, XGBoost, KAN) and sequence tensors (LSTM)
  5. Train all 4 models
  6. Evaluate + compare on the held-out test set
  7. Save a comparison table + plots to outputs/

Run with:  python -m src.main
(from the project root, after `pip install -r requirements.txt`)
"""
import argparse
import os
import sys

import numpy as np

# Allow running as `python src/main.py` as well as `python -m src.main`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_raw_data
from src.feature_engineering import (
    clean_and_engineer, impute_remaining, patient_level_split,
    get_tabular_feature_columns, build_sequences,
)
from src.models.tree_models import train_random_forest, train_xgboost
from src.models.deep_models import train_lstm, train_kan, predict_proba_torch
from src.evaluate import build_comparison_report


def parse_args():
    parser = argparse.ArgumentParser(description="Sepsis early-prediction model comparison")
    parser.add_argument("--raw-dir", type=str, default=config.RAW_DATA_DIR,
                         help="Folder containing the extracted PhysioNet training_setA/B data")
    parser.add_argument("--max-patients", type=int, default=config.MAX_PATIENTS,
                         help="Subsample N patients for a quick test run (default: use all)")
    parser.add_argument("--force-reload", action="store_true",
                         help="Ignore the cached consolidated CSV and re-parse raw .psv files")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training (faster)")
    parser.add_argument("--skip-kan", action="store_true", help="Skip KAN training (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(config.RANDOM_SEED)

    # ---------------- 1. Load ----------------
    df = load_raw_data(raw_dir=args.raw_dir, max_patients=args.max_patients,
                        force_reload=args.force_reload)

    # ---------------- 2. Feature engineering ----------------
    print("[main] Engineering features (missingness flags, forward-fill, rolling stats)...")
    df = clean_and_engineer(df)

    # ---------------- 3. Patient-level split ----------------
    train_df, val_df, test_df = patient_level_split(df)

    tabular_cols = get_tabular_feature_columns(df)
    print(f"[main] Using {len(tabular_cols)} tabular features: {tabular_cols}")

    train_df, (val_df, test_df), medians = impute_remaining(
        train_df, [val_df, test_df], tabular_cols
    )

    X_train_tab = train_df[tabular_cols].to_numpy(dtype=np.float32)
    y_train = train_df["SepsisLabel"].to_numpy(dtype=np.float32)
    X_val_tab = val_df[tabular_cols].to_numpy(dtype=np.float32)
    y_val = val_df["SepsisLabel"].to_numpy(dtype=np.float32)
    X_test_tab = test_df[tabular_cols].to_numpy(dtype=np.float32)
    y_test = test_df["SepsisLabel"].to_numpy(dtype=np.float32)

    all_predictions = {}
    all_labels = {}

    # ---------------- 4a. Random Forest ----------------
    print("\n[main] Training Random Forest...")
    rf_model = train_random_forest(X_train_tab, y_train)
    rf_proba = rf_model.predict_proba(X_test_tab)[:, 1]
    all_predictions["RandomForest"] = rf_proba
    all_labels["RandomForest"] = y_test

    # ---------------- 4b. XGBoost ----------------
    print("\n[main] Training XGBoost...")
    try:
        xgb_model = train_xgboost(X_train_tab, y_train)
        xgb_proba = xgb_model.predict_proba(X_test_tab)[:, 1]
        all_predictions["XGBoost"] = xgb_proba
        all_labels["XGBoost"] = y_test
    except ImportError:
        print("[main] xgboost not installed - skipping. Run: pip install xgboost")

    # ---------------- 4c. KAN (tabular deep learning) ----------------
    if not args.skip_kan:
        print("\n[main] Training KAN...")
        try:
            kan_model = train_kan(X_train_tab, y_train, X_val_tab, y_val,
                                   n_features=X_train_tab.shape[1])
            kan_proba = predict_proba_torch(kan_model, X_test_tab)
            all_predictions["KAN"] = kan_proba
            all_labels["KAN"] = y_test
        except ImportError:
            print("[main] torch not installed - skipping KAN. Run: pip install torch")

    # ---------------- 4d. LSTM (sequential deep learning) ----------------
    if not args.skip_lstm:
        print("\n[main] Building sequences for LSTM...")
        X_train_seq, y_train_seq = build_sequences(train_df)
        X_val_seq, y_val_seq = build_sequences(val_df)
        X_test_seq, y_test_seq = build_sequences(test_df)

        print("[main] Training LSTM...")
        try:
            lstm_model = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                     n_features=X_train_seq.shape[2])
            lstm_proba = predict_proba_torch(lstm_model, X_test_seq)
            all_predictions["LSTM"] = lstm_proba
            all_labels["LSTM"] = y_test_seq
        except ImportError:
            print("[main] torch not installed - skipping LSTM. Run: pip install torch")

    # ---------------- 5. Compare ----------------
    if len(all_predictions) == 0:
        print("[main] No models were trained successfully - check installed packages.")
        return

    build_comparison_report(all_predictions, all_labels)


if __name__ == "__main__":
    main()
