"""
classification.py
-----------------
Customer classification by heating system type using load–temperature behavior.
Mirrors: slope_120_meters-aggregate.ipynb, Electri_vs_bin_plots.ipynb

Pipeline:
  1. Fit piecewise linear regression per meter on load–temperature scatter
  2. Compute heating-side slope (< 35°F region)
  3. Bin heat pump meters by slope range
  4. Score unlabeled meters by MAE vs. heat pump forecast
  5. Classify as: Heat Pump / Electricity / Gas

Data structure (df_2024 from slope_120_meters-aggregate.ipynb):
    columns: asset_id, timestamp, date, hour, value, avg_temp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ── Piecewise temperature breakpoints (°F) ───────────────────────────────────
BREAKPOINTS = [35, 65, 80]


# ── Step 1: Piecewise regression slope per meter ─────────────────────────────

def piecewise_slope(temps: np.ndarray, loads: np.ndarray,
                    bp_low: float = 35.0) -> float:
    """
    Compute the OLS slope of load on temperature for observations
    below the heating-side breakpoint (< bp_low °F).

    Returns the slope coefficient (negative = more load at lower temps = heating signal).
    """
    mask = temps < bp_low
    if mask.sum() < 10:
        return np.nan
    model = LinearRegression()
    model.fit(temps[mask].reshape(-1, 1), loads[mask])
    return model.coef_[0]


def compute_meter_slopes(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    temp_col: str = "avg_temp",
    load_col: str = "value",
    bp_low: float = 35.0,
    bp_high: float = 80.0,
) -> pd.DataFrame:
    """
    Compute heating-side and cooling-side slopes for every meter.

    Returns DataFrame with columns: asset_id, slope_heating, slope_cooling
    """
    records = []
    for asset, grp in df.groupby(asset_col):
        temps = grp[temp_col].values
        loads = grp[load_col].values
        s_heat = piecewise_slope(temps, loads, bp_low=bp_low)
        s_cool = piecewise_slope(-temps, loads, bp_low=-bp_high)  # cooling: above bp_high
        records.append({"asset_id": asset, "slope_heating": s_heat, "slope_cooling": -s_cool})

    return pd.DataFrame(records)


# ── Step 2: Slope-based binning ───────────────────────────────────────────────

BIN_DEFINITIONS = [
    ("Bin_1", 0.0,    -0.049),
    ("Bin_2", -0.05,  -0.099),
    ("Bin_3", -0.10,  -0.149),
    ("Bin_4", -0.15,  -np.inf),
]


def assign_bins(slopes_df: pd.DataFrame, slope_col: str = "slope_heating") -> pd.DataFrame:
    """
    Assign each heat pump meter to a bin based on its heating-side slope.

    Bin 1: 0 to -0.049  (low usage / barely heating-responsive)
    Bin 2: -0.05 to -0.099  (average usage — typical heat pump)
    Bin 3: -0.10 to -0.149  (above-average usage — typical heat pump)
    Bin 4: -0.15 and below  (high usage, similar to electric resistance)
    """
    df = slopes_df.copy()

    def _bin(slope):
        if pd.isna(slope):
            return "Unknown"
        for name, hi, lo in BIN_DEFINITIONS:
            if lo <= slope <= hi:
                return name
        return "Bin_4"

    df["bin"] = df[slope_col].apply(_bin)
    print(df["bin"].value_counts().sort_index())
    return df


# ── Step 3: MAE-based classification of unlabeled meters ─────────────────────

def classify_unlabeled_meters(
    df_unlabeled: pd.DataFrame,
    df_hp_forecast: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    load_col: str = "value",
    forecast_col: str = "hp_forecast",
    mae_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Compare unlabeled meter actual load against heat pump forecast.
    High normalized MAE → not a heat pump.

    Parameters
    ----------
    df_unlabeled  : actual load of unlabeled meters
    df_hp_forecast: forecast generated from 98 labeled heat pump meters
                    joined on timestamp
    mae_threshold : normalized MAE above this → candidate for non-HP classification

    Returns
    -------
    DataFrame with columns: asset_id, norm_mae, candidate_class
    """
    merged = df_unlabeled.merge(df_hp_forecast[[timestamp_col, forecast_col]],
                                 on=timestamp_col, how="inner")

    results = []
    for asset, grp in merged.groupby(asset_col):
        avg_load = grp[load_col].mean()
        if avg_load == 0:
            continue
        mae_val = mean_absolute_error(grp[load_col], grp[forecast_col])
        norm_mae = mae_val / avg_load
        results.append({"asset_id": asset, "norm_mae": norm_mae, "avg_load": avg_load})

    df_result = pd.DataFrame(results).sort_values("norm_mae", ascending=False)
    df_result["candidate_class"] = np.where(
        df_result["norm_mae"] > mae_threshold, "non_heat_pump", "heat_pump"
    )
    return df_result


def classify_by_seasonal_pattern(
    df: pd.DataFrame,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    load_col: str = "value",
) -> pd.DataFrame:
    """
    Secondary classification: split non-HP meters into Electricity vs Gas
    by comparing winter vs summer load ratio.

    Electric heating: high winter, lower summer  (ratio > 1.2)
    Gas heating:      low winter electricity, higher summer (ratio < 0.8)
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["month"] = df[timestamp_col].dt.month

    winter = df[df["month"].isin([12, 1, 2])]
    summer = df[df["month"].isin([6, 7, 8])]

    w_avg = winter.groupby(asset_col)[load_col].mean().rename("winter_avg")
    s_avg = summer.groupby(asset_col)[load_col].mean().rename("summer_avg")

    ratio_df = pd.concat([w_avg, s_avg], axis=1)
    ratio_df["winter_summer_ratio"] = ratio_df["winter_avg"] / ratio_df["summer_avg"]

    def _class(ratio):
        if ratio > 1.2:
            return "electricity"
        elif ratio < 0.8:
            return "gas"
        else:
            return "heat_pump"

    ratio_df["heating_class"] = ratio_df["winter_summer_ratio"].apply(_class)
    return ratio_df.reset_index()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_load_vs_temp(
    df: pd.DataFrame,
    class_col: str = "heating_class",
    temp_col: str = "avg_temp",
    load_col: str = "value",
    breakpoints: list[float] = BREAKPOINTS,
    title: str = "Load vs Temperature by Customer Class",
    save_path: str | None = None,
):
    """
    Reproduce the three-class load vs temperature scatter with piecewise slopes.
    Mirrors Figure 0-17 from the capstone report.
    """
    colors = {"heat_pump": "green", "electricity": "cyan", "gas": "orange",
              "Bin_1": "gray", "Bin_2": "red", "Bin_3": "olive", "Bin_4": "purple"}

    fig, ax = plt.subplots(figsize=(12, 6))

    for cls, grp in df.groupby(class_col):
        color = colors.get(cls, "blue")
        ax.scatter(grp[temp_col], grp[load_col], s=3, alpha=0.4,
                   color=color, label=cls)

    for bp in breakpoints:
        ax.axvline(bp, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Temperature °F")
    ax.set_ylabel("Predicted Load kW")
    ax.set_title(title)
    ax.legend(markerscale=4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    plt.show()
