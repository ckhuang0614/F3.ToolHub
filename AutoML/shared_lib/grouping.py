from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple


def make_group_id(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group_key columns missing in CSV: {missing}")
    return df[group_cols].astype(str).agg("|".join, axis=1)


def group_shuffle_split(
    df: pd.DataFrame,
    group_id: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size out of range")
    rng = np.random.default_rng(random_seed)
    groups = group_id.unique()
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(groups[:n_test])
    is_test = group_id.isin(test_groups)
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)
