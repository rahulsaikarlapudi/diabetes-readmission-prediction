"""Load the Diabetes 130-US Hospitals dataset (1999-2008).

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("..")
    from src.data_loader import load_raw_data, load_id_mappings, add_target
"""

from pathlib import Path

import pandas as pd

# Repo root = the folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(path: Path = RAW_DIR / "diabetic_data.csv") -> pd.DataFrame:
    """Read diabetic_data.csv. The dataset uses '?' for missing values.

    keep_default_na=False matters: in max_glu_serum / A1Cresult the string "None"
    means "test not performed" (a real category), but pandas would treat it as NaN.
    """
    return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)


def load_id_mappings(path: Path = RAW_DIR / "IDS_mapping.csv") -> dict[str, pd.DataFrame]:
    """Split IDS_mapping.csv into its three lookup tables.

    The file stacks three tables (admission_type_id, discharge_disposition_id,
    admission_source_id), separated by blank lines, each with its own header row.
    Returns {"admission_type_id": df, ...} with columns [<id column>, "description"].
    """
    tables: dict[str, pd.DataFrame] = {}
    rows: list[list[str]] = []
    id_col = None

    for line in path.read_text().splitlines() + [","]:
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts[0]:  # blank separator line -> close the current table
            if id_col:
                tables[id_col] = pd.DataFrame(rows, columns=[id_col, "description"]).astype({id_col: int})
            id_col, rows = None, []
        elif id_col is None:  # first line of a block is its header
            id_col = parts[0]
        else:
            rows.append(parts)

    return tables


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target: 1 if readmitted within 30 days ('<30'), else 0."""
    df = df.copy()
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    return df
