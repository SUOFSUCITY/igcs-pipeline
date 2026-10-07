"""Validate and export location probabilities."""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.config import COL_ID, LOCATION_COLS_MAP


def extract_location_encoding(csv_path, output_path, split_name, *, id_column="id"):
    import numpy as np
    import pandas as pd

    columns = list(LOCATION_COLS_MAP)
    if id_column in columns:
        raise ValueError("The identifier column must differ from probability columns.")
    frame = pd.read_csv(csv_path, dtype={id_column: "string"})
    required = [id_column, *columns]
    if any(column not in frame for column in required):
        raise ValueError("Input does not match the public probability schema.")
    ids = frame[id_column]
    if (
        frame.empty
        or ids.isna().any()
        or ids.str.strip().eq("").any()
        or ids.duplicated().any()
    ):
        raise ValueError(
            "Provide a nonempty table with unique, nonempty case identifiers."
        )
    probabilities = frame[columns].apply(pd.to_numeric, errors="raise")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    result = probabilities.rename(columns=LOCATION_COLS_MAP)
    result.insert(0, COL_ID, ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    print(f"{split_name}: exported {len(result)} rows.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()
    extract_location_encoding(
        args.input, args.output, "Input", id_column=args.id_column
    )


if __name__ == "__main__":
    main()
