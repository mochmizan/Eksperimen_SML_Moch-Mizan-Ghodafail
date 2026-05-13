"""
automate_Moch-Mizan-Ghodafail.py
Script untuk melakukan preprocessing data breast cancer secara otomatis.

Cara menjalankan (dari dalam folder preprocessing/):
    python automate_Moch-Mizan-Ghodafail.py

Output:
    cancer_preprocessing/train.csv
    cancer_preprocessing/test.csv
"""

import pandas as pd
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ─── Path Configuration ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # folder preprocessing/
RAW_DATA    = os.path.join(BASE_DIR, '..', 'breast_cancer_raw.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'cancer_preprocessing')
FEATURE_COLS = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
    'mean_smoothness', 'mean_compactness', 'mean_concavity',
    'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
    'radius_error', 'texture_error', 'perimeter_error', 'area_error',
    'smoothness_error', 'compactness_error', 'concavity_error',
    'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
    'worst_smoothness', 'worst_compactness', 'worst_concavity',
    'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]
TARGET_COL  = 'target'


# ─── Functions ─────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File tidak ditemukan: {filepath}\n"
            "Jalankan 'python generate_raw_data.py' terlebih dahulu."
        )
    df = pd.read_csv(filepath)
    print(f"[load_data] Data dimuat. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Isi missing values numerik dengan median."""
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  [missing] {col}: diisi median = {median_val:.4f}")
    print(f"[missing_values] Total missing: {total_missing} -> 0")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after  = len(df)
    print(f"[duplicates] Dihapus: {before - after} baris. Sisa: {after}")
    return df


def encode_target(df: pd.DataFrame, target_col: str):
    """Label encoding pada kolom target. Return df dan encoder."""
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    print(f"[encode] Mapping: { {cls: i for i, cls in enumerate(le.classes_)} }")
    return df, le


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler fit pada train, transform pada keduanya."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    print(f"[scaling] StandardScaler applied. Mean train ~ {X_train_scaled.mean().mean():.4f}")
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Train-test split dengan stratify."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def save_data(
    X_train, X_test, y_train, y_test,
    feature_cols: list, target_col: str, output_dir: str
):
    """Simpan train dan test CSV ke output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1
    )
    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1
    )

    train_path = os.path.join(output_dir, 'train.csv')
    test_path  = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[save] train.csv  -> {train_path} | Shape: {train_df.shape}")
    print(f"[save] test.csv   -> {test_path}  | Shape: {test_df.shape}")
    return train_path, test_path


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing():
    print("=" * 55)
    print("  PIPELINE PREPROCESSING BREAST CANCER - Moch Mizan Ghodafail")
    print("=" * 55)

    # 1. Load
    df = load_data(RAW_DATA)

    # 2. Handle missing values
    df = handle_missing_values(df, FEATURE_COLS)

    # 3. Handle duplicates
    df = handle_duplicates(df)

    # 4. Encode target
    df, le = encode_target(df, TARGET_COL)

    # 5. Separate features & target
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 6. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 7. Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # 8. Save
    save_data(X_train_s, X_test_s, y_train, y_test,
              FEATURE_COLS, TARGET_COL, OUTPUT_DIR)

    print("=" * 55)
    print("  PREPROCESSING SELESAI!")
    print("=" * 55)
    return X_train_s, X_test_s, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
