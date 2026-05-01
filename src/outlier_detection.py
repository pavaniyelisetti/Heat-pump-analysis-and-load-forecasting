"""
outlier_detection.py
--------------------
Iterative APE-based outlier detection for smart meter data cleansing.
Mirrors: insample_outlierdetect-recency+holiday.ipynb

Method (Xie, 2016 adapted):
  1. Fit Vanilla + Recency + Holiday model in-sample on full data
  2. Compute APE for every observation
  3. Sort timestamps by APE descending
  4. Remove top N% (thresholds: 100% → 80% → 60% → 50% → 40%)
  5. Repeat until convergence

Usage
-----
    from outlier_detection import run_outlier_detection
    df_clean, excluded = run_outlier_detection(df_meter)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vanilla_benchmark import VanillaBenchmarkModel


# APE thresholds applied sequentially (percentage of max APE)
DEFAULT_THRESHOLDS = [100, 80, 60, 50, 40]


def compute_ape(df: pd.DataFrame, model: VanillaBenchmarkModel,
                load_col: str = "value") -> pd.Series:
    """Return APE series indexed to match df."""
    return model.ape(df, load_col=load_col)


def plot_ape_curve(ape_series: pd.Series, threshold_line: float | None = None,
                   title: str = "APE over observations"):
    """Plot sorted APE values to help select removal thresholds."""
    sorted_ape = ape_series.sort_values(ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.plot(sorted_ape.values)
    if threshold_line:
        plt.axhline(threshold_line, color="red", linestyle="--", label=f"Threshold = {threshold_line}")
        plt.legend()
    plt.xlabel("Observation rank (sorted by APE)")
    plt.ylabel("APE (%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def run_outlier_detection(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    load_col: str = "value",
    thresholds: list[int] = DEFAULT_THRESHOLDS,
    n_remove_per_pass: int = 100,
    use_recency: bool = True,
    use_holiday: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run iterative APE-based outlier detection on a single meter's data.

    Parameters
    ----------
    df               : DataFrame for ONE meter (or aggregate), sorted by timestamp
    thresholds       : APE percentile thresholds applied in each pass
    n_remove_per_pass: max observations to remove per pass
    use_recency      : include recency effect in model
    use_holiday      : include holiday effect in model
    verbose          : print progress

    Returns
    -------
    df_clean   : DataFrame with outliers replaced by predicted values
    excluded   : DataFrame of all excluded observations with their APE values
    """
    df_work = df.copy()
    df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
    excluded_dfs = []

    for threshold in thresholds:
        model = VanillaBenchmarkModel(use_recency=use_recency, use_holiday=use_holiday)
        model.fit(df_work)

        ape = model.ape(df_work, load_col=load_col)

        # Find top n_remove_per_pass observations above APE threshold percentile
        cutoff = np.percentile(ape.dropna(), threshold)
        to_remove_idx = ape[ape >= cutoff].nlargest(n_remove_per_pass).index

        if len(to_remove_idx) == 0:
            break

        # Record excluded rows
        excl = df_work.loc[to_remove_idx].copy()
        excl["APE"] = ape[to_remove_idx]
        excl["threshold_pass"] = threshold
        excluded_dfs.append(excl)

        # Replace outlier values with model predictions
        pred_vals = model.predict(df_work.loc[to_remove_idx])
        df_work.loc[to_remove_idx, load_col] = pred_vals

        if verbose:
            print(f"  Pass threshold={threshold}%: removed/replaced {len(to_remove_idx)} "
                  f"observations (max APE: {ape.max():.1f}%)")

    excluded_all = pd.concat(excluded_dfs, axis=0).sort_values(timestamp_col) if excluded_dfs else pd.DataFrame()

    if verbose:
        print(f"\nTotal outliers detected and replaced: {len(excluded_all)}")

    return df_work, excluded_all


def run_outlier_detection_all_meters(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    load_col: str = "value",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run outlier detection on every meter independently.

    Returns
    -------
    df_all_clean  : full cleaned DataFrame (all meters concatenated)
    excluded_all  : all excluded observations across all meters
    """
    clean_parts = []
    excluded_parts = []

    meters = df[asset_col].unique()
    for i, meter in enumerate(meters):
        df_m = df[df[asset_col] == meter].copy()
        try:
            clean_m, excl_m = run_outlier_detection(df_m, asset_col, timestamp_col,
                                                      load_col, verbose=False, **kwargs)
            clean_parts.append(clean_m)
            if len(excl_m):
                excluded_parts.append(excl_m)
        except Exception as e:
            print(f"  Skipped meter {meter}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(meters)} meters...")

    df_clean = pd.concat(clean_parts, axis=0).reset_index(drop=True)
    excluded = pd.concat(excluded_parts, axis=0).reset_index(drop=True) if excluded_parts else pd.DataFrame()
    print(f"\nDone. Total outliers replaced across {len(meters)} meters: {len(excluded)}")
    return df_clean, excluded
