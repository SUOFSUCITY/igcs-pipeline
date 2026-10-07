"""Feature assembly for the conservative-treatment cohort."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import parameter, validate_parameters

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from features.config import (
    FEATURES_DIR,
    CLINICAL_TRAIN_CSV,
    CLINICAL_TEST_CSV,
    COL_ID,
    COL_SURGERY,
    COL_PRIMARY_ENDPOINT,
    COL_SECONDARY_ENDPOINT,
    CLINICAL_COLS_MAP,
    LOCATION_LABEL_COL,
    LOCATION_LABEL_MAP,
    INTERACTION_PAIRS,
    LOCATION_COLS,
    IMAGING_ALL_COLS,
    PASH_COLS,
    CLINICAL_FEATURE_COLS,
    GCS_NONLINEAR_COLS,
)


def load_clinical(csv_path):
    df = pd.read_csv(csv_path, dtype={COL_ID: str})
    df = df.rename(columns=CLINICAL_COLS_MAP)
    keep_cols = [
        COL_ID,
        COL_SURGERY,
        LOCATION_LABEL_COL,
        COL_PRIMARY_ENDPOINT,
        COL_SECONDARY_ENDPOINT,
    ]
    keep_cols += list(CLINICAL_COLS_MAP.values())
    df = df[keep_cols]
    return df


def build_interaction_features(df):
    for col_a, col_b in INTERACTION_PAIRS:
        if col_a in df.columns and col_b in df.columns:
            feat_name = f"inter_{col_a}_{col_b}"
            df[feat_name] = df[col_a] * df[col_b]
    return df


def build_for_split(clinical_csv, split_name, suffix):
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
    df_clinical = load_clinical(clinical_csv)
    n_total = len(df_clinical)
    n_surgery = int(df_clinical[COL_SURGERY].sum())
    df_clinical = df_clinical[df_clinical[COL_SURGERY] == 0].copy()
    n_after = len(df_clinical)
    df_loc = pd.read_csv(
        FEATURES_DIR / f"location_encoding_{suffix}.csv", dtype={COL_ID: str}
    )
    df_img = pd.read_csv(
        FEATURES_DIR / f"imaging_features_{suffix}.csv", dtype={COL_ID: str}
    )
    df_pash = pd.read_csv(
        FEATURES_DIR / f"pash_features_{suffix}.csv", dtype={COL_ID: str}
    )
    df = df_clinical.merge(df_loc, on=COL_ID, how="left")
    df = df.merge(df_img, on=COL_ID, how="left")
    df = df.merge(df_pash, on=COL_ID, how="left")
    n_missing_loc = df[LOCATION_COLS[0]].isna().sum()
    n_missing_img = (
        df[IMAGING_ALL_COLS[0]].isna().sum()
        if IMAGING_ALL_COLS[0] in df.columns
        else -1
    )
    n_missing_pash = df[PASH_COLS[0]].isna().sum() if PASH_COLS[0] in df.columns else -1
    feature_cols = LOCATION_COLS + IMAGING_ALL_COLS + PASH_COLS + CLINICAL_FEATURE_COLS
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
    df = build_interaction_features(df)
    interaction_cols = [f"inter_{a}_{b}" for a, b in INTERACTION_PAIRS]
    df["GCS_le8"] = (df["GCS"] <= 8).astype(int)
    df["GCS_le12"] = (df["GCS"] <= 12).astype(int)
    for lbl, name in LOCATION_LABEL_MAP.items():
        df[f"loc_onehot_{name}"] = (df[LOCATION_LABEL_COL] == lbl).astype(int)
    onehot_cols = [f"loc_onehot_{name}" for name in LOCATION_LABEL_MAP.values()]
    all_feature_cols = (
        LOCATION_COLS
        + IMAGING_ALL_COLS
        + PASH_COLS
        + CLINICAL_FEATURE_COLS
        + interaction_cols
        + GCS_NONLINEAR_COLS
        + onehot_cols
    )
    meta_cols = [
        COL_ID,
        LOCATION_LABEL_COL,
        COL_PRIMARY_ENDPOINT,
        COL_SECONDARY_ENDPOINT,
    ]
    output_cols = meta_cols + all_feature_cols
    df_out = df[output_cols]
    print(f"\n{'=' * 40}")
    print(f"{'=' * 40}")
    y_primary = df_out[COL_PRIMARY_ENDPOINT]
    y_secondary = df_out[COL_SECONDARY_ENDPOINT]
    fully_missing = [c for c in all_feature_cols if df_out[c].isna().all()]
    if fully_missing:
        pass
    remaining_na = df_out[all_feature_cols].isna().sum().sum()
    return (df_out, output_cols)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df_train, cols = build_for_split(CLINICAL_TRAIN_CSV, "train", "train")
    train_out = FEATURES_DIR / "feature_matrix_train.csv"
    df_train.to_csv(train_out, index=False)
    df_test, _ = build_for_split(CLINICAL_TEST_CSV, "test", "test")
    test_out = FEATURES_DIR / "feature_matrix_test.csv"
    df_test.to_csv(test_out, index=False)
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
