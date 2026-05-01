"""
cleansing.py
------------
Smart meter data cleansing pipeline for heat pump load analysis.

Steps:
  1. Heatmap-based bad meter detection (missing/zero values)
  2. Natural event flagging — zero readings during verified extreme weather
  3. Abnormal consumption behavior detection (high/flat, sudden lifts, pool pumps)
  4. Group-based MAPE screening for remaining irregular meters

References: Wang et al. (2018), Section 4.2 of capstone report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ── Natural events to preserve (not remove) ──────────────────────────────────
# Zero-load periods correlated with verified extreme weather or outages.
# These are KEPT as valid observations.

NATURAL_EVENTS = [
    {"start": "2022-01-22 20:00:00", "end": "2022-01-23 00:00:00",
     "description": "Winter storm Jan 22-23"},
    {"start": "2022-03-19 21:00:00", "end": "2022-03-20 00:00:00",
     "description": "Lowest negative temperatures of the month (-32.4°C)"},
    {"start": "2023-05-18 21:00:00", "end": "2023-05-19 20:00:00",
     "description": "Flash flooding Brunswick/New Hanover Counties"},
    {"start": "2023-07-20 21:00:00", "end": "2023-07-21 20:00:00",
     "description": "Highest temperature of month (35.1°C)"},
    {"start": "2023-11-30 20:00:00", "end": "2023-12-01 19:00:00",
     "description": "Lowest temperature of month, sub-zero start"},
]


def flag_natural_events(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Mark rows that fall within a known natural event period."""
    df = df.copy()
    df["natural_event"] = False
    for event in NATURAL_EVENTS:
        mask = (df[timestamp_col] >= event["start"]) & (df[timestamp_col] <= event["end"])
        df.loc[mask, "natural_event"] = True
    return df


# ── Step 1: Heatmap bad meter detection ──────────────────────────────────────

def plot_meter_heatmap(df: pd.DataFrame, asset_col: str = "asset_id",
                       timestamp_col: str = "timestamp", value_col: str = "value",
                       save_path: str | None = None):
    """
    Visualize all meters as a heatmap:
      white = missing, red = zero, black = non-zero numeric value.

    Parameters
    ----------
    df         : long-format DataFrame with one row per (meter, timestamp)
    asset_col  : column identifying each meter
    save_path  : if provided, save the figure here
    """
    pivot = df.pivot_table(index=asset_col, columns=timestamp_col,
                           values=value_col, aggfunc="first")

    status = np.where(pivot.isna(), 0,            # white: missing
              np.where(pivot == 0, 1,              # red: zero
                       2))                         # black: numeric

    cmap = mcolors.ListedColormap(["white", "red", "black"])

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(status, aspect="auto", cmap=cmap, interpolation="none")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Meter (asset_id)")
    ax.set_title("Heatmap: Missing (White), Zero (Red), Non-Zero (Black)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved to {save_path}")
    plt.show()


def remove_bad_meters(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    zero_threshold: float = 0.20,
    missing_threshold: float = 0.20,
) -> tuple[pd.DataFrame, list]:
    """
    Remove meters with excessive zero or missing values.

    Parameters
    ----------
    zero_threshold    : fraction of zero readings above which a meter is dropped
    missing_threshold : fraction of missing readings above which a meter is dropped

    Returns
    -------
    (cleaned_df, list_of_removed_asset_ids)
    """
    grouped = df.groupby(asset_col)[value_col]
    total_counts = grouped.count() + grouped.apply(lambda x: x.isna().sum())
    zero_frac   = grouped.apply(lambda x: (x == 0).sum()) / total_counts
    miss_frac   = grouped.apply(lambda x: x.isna().sum()) / total_counts

    bad = zero_frac[zero_frac > zero_threshold].index.tolist()
    bad += miss_frac[miss_frac > missing_threshold].index.tolist()
    bad = list(set(bad))

    cleaned = df[~df[asset_col].isin(bad)].copy()
    print(f"Removed {len(bad)} bad meters. Remaining: {cleaned[asset_col].nunique()}")
    return cleaned, bad


# ── Step 2: Abnormal behavior detection ──────────────────────────────────────

def detect_abnormal_meters(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    avg_load_col: str | None = None,
    high_flat_threshold: float = 3.0,   # ratio to median across all meters
    cv_threshold: float = 0.20,         # low CV = suspiciously flat
) -> list:
    """
    Detect meters with abnormal consumption patterns:
      - Persistently high flat load (potential mining/industrial)
      - Sudden sustained lift partway through the year
      - Extremely high coefficient of variation (pool pumps, etc.)

    Returns list of asset_ids to investigate/remove.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    suspect = []
    median_overall = df[value_col].median()

    for asset, group in df.groupby(asset_col):
        avg = group[value_col].mean()
        std = group[value_col].std()
        cv  = std / avg if avg > 0 else 0

        # High flat consumption
        if avg > high_flat_threshold * median_overall and cv < cv_threshold:
            suspect.append(asset)
            continue

        # Sudden lift: compare first-half vs second-half
        mid = len(group) // 2
        first_half_avg  = group[value_col].iloc[:mid].mean()
        second_half_avg = group[value_col].iloc[mid:].mean()
        if first_half_avg > 0 and second_half_avg / first_half_avg > 2.5:
            suspect.append(asset)

    return list(set(suspect))


# ── Step 3: MAPE group screening ─────────────────────────────────────────────

def screen_by_mape(
    df_mape: pd.DataFrame,
    asset_col: str = "asset_id",
    mape_col: str = "MAPE",
    mape_threshold: float = 50.0,
) -> list:
    """
    Remove meters whose per-meter MAPE exceeds the threshold.
    Typically applied after group-level forecasting to catch group 4 / group 7 outliers.

    Parameters
    ----------
    df_mape        : DataFrame with one row per meter and a MAPE column
    mape_threshold : meters above this MAPE are flagged

    Returns
    -------
    list of asset_ids to remove
    """
    bad = df_mape[df_mape[mape_col] > mape_threshold][asset_col].tolist()
    print(f"MAPE screening: flagged {len(bad)} meters above {mape_threshold}% MAPE")
    return bad


# ── Full cleansing pipeline ───────────────────────────────────────────────────

def run_cleansing_pipeline(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    save_heatmap: str | None = "results/figures/heatmap.png",
) -> pd.DataFrame:
    """
    Run the full sequential cleansing pipeline and return a clean DataFrame.

    Steps:
      1. Flag natural events (retain these zeros)
      2. Plot heatmap for visual inspection
      3. Remove bad meters (missing/zero threshold)
      4. Detect and remove abnormal consumption behaviors
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = flag_natural_events(df, timestamp_col)

    print(f"Starting meters: {df[asset_col].nunique()}")

    # Step 1: Heatmap + bad meter removal
    if save_heatmap:
        plot_meter_heatmap(df, asset_col, timestamp_col, value_col, save_path=save_heatmap)
    df, bad_meters = remove_bad_meters(df, asset_col, timestamp_col, value_col)

    # Step 2: Abnormal behavior removal
    abnormal = detect_abnormal_meters(df, asset_col, timestamp_col, value_col)
    df = df[~df[asset_col].isin(abnormal)].copy()
    print(f"Removed {len(abnormal)} abnormal-behavior meters. Remaining: {df[asset_col].nunique()}")

    print(f"\nFinal meter count after cleansing: {df[asset_col].nunique()}")
    return df
