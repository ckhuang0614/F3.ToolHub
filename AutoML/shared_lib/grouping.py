from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_group_id(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    """Create a group id series.

    Fallback behavior:
    - If group_cols is empty / None: use row index as group id (row-level split).
      This prevents accidental degenerate splits where all rows fall into a single group.
    """
    if not group_cols:
        # row-level grouping (unique per row)
        return df.index.astype(str)

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group_key columns missing in CSV: {missing}")
    return df[group_cols].astype(str).agg("|".join, axis=1)


def row_shuffle_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level random split.

    Always guarantees at least 1 row in both train and test when len(df) >= 2.
    """
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size out of range")

    n = int(len(df))
    if n <= 1:
        # Not enough rows to split; return all rows to train.
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = max(1, int(round(n * float(test_size))))
    n_test = min(n - 1, n_test)  # keep at least 1 row in train

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def group_shuffle_split(
    df: pd.DataFrame,
    group_id: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size out of range")
    rng = np.random.default_rng(random_seed)
    group_values = pd.Index(group_id)
    groups = group_values.unique().to_numpy()

    # Degenerate case: all rows are in a single group.
    # Fallback to row-level split (otherwise train set becomes empty).
    if len(groups) <= 1:
        return row_shuffle_split(df, test_size=test_size, random_seed=random_seed)

    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_size)))
    n_test = min(len(groups) - 1, n_test)  # keep at least 1 group in train
    test_groups = set(groups[:n_test])
    is_test = group_values.isin(test_groups)
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)
