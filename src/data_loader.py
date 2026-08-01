"""
Loads and consolidates the PhysioNet/CinC 2019 Sepsis Challenge dataset.

The raw data ships as one .psv (pipe-separated) file per patient, spread
across training_setA/ and training_setB/ folders. This module finds every
.psv file under RAW_DATA_DIR (any depth), reads it, tags it with a
PatientID and Hospital source, and concatenates everything into a single
tidy DataFrame. The result is cached to disk so repeat runs are fast.
"""
import os
import glob
import pandas as pd

import config


def _find_psv_files(root_dir):
    pattern = os.path.join(root_dir, "**", "*.psv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under '{root_dir}'. "
            f"Make sure you extracted the PhysioNet archive into this folder "
            f"(it should contain training_setA / training_setB subfolders)."
        )
    return sorted(files)


def _infer_hospital(filepath):
    """training_setA -> hospital 'A', training_setB -> hospital 'B'."""
    lower = filepath.lower()
    if "training_seta" in lower or os.sep + "a" + os.sep in lower:
        return "A"
    if "training_setb" in lower or os.sep + "b" + os.sep in lower:
        return "B"
    return "unknown"


def load_raw_data(raw_dir=None, max_patients=None, force_reload=False):
    """
    Returns a single consolidated DataFrame with one row per patient-hour,
    plus PatientID and Hospital columns.
    """
    raw_dir = raw_dir or config.RAW_DATA_DIR
    cache_path = config.PROCESSED_DATA_PATH

    if os.path.exists(cache_path) and not force_reload:
        print(f"[data_loader] Loading cached consolidated dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        if max_patients is not None:
            keep_ids = df["PatientID"].drop_duplicates().iloc[:max_patients]
            df = df[df["PatientID"].isin(keep_ids)].reset_index(drop=True)
        return df

    print(f"[data_loader] Consolidating raw .psv files from {raw_dir} ...")
    files = _find_psv_files(raw_dir)
    print(f"[data_loader] Found {len(files)} patient files.")

    if max_patients is not None:
        files = files[:max_patients]
        print(f"[data_loader] Subsampled to {len(files)} patient files (MAX_PATIENTS set).")

    frames = []
    for i, f in enumerate(files):
        pid = os.path.splitext(os.path.basename(f))[0]
        d = pd.read_csv(f, sep="|")
        d["PatientID"] = pid
        d["Hospital"] = _infer_hospital(f)
        frames.append(d)
        if (i + 1) % 5000 == 0:
            print(f"[data_loader]   ...{i + 1}/{len(files)} files parsed")

    df = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Consolidated shape: {df.shape}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[data_loader] Cached consolidated dataset to {cache_path}")

    return df


if __name__ == "__main__":
    data = load_raw_data(max_patients=config.MAX_PATIENTS)
    print(data.head())
    print(data.shape)
