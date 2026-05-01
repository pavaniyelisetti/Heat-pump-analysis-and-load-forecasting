"""
weather.py
----------
Weather station averaging and temperature feature engineering.

Combines 4 Brunswick County meteorological stations into a single
averaged hourly temperature series, then builds lag and moving average
features used by the Vanilla Benchmark and Recency Effect models.

Data structure expected:
    columns: station, est (datetime string), date, hour, temp, rel_humidity,
             heat_index, pressure_millibars, visibility, wind_speed, wind_chill
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Station averaging ────────────────────────────────────────────────────────

def load_weather(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load CSVs from multiple weather stations and return a single DataFrame
    with averaged temperature (avg_temp) per timestamp.

    Parameters
    ----------
    paths : list of file paths, one per station

    Returns
    -------
    DataFrame with columns: datetime, avg_temp
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"].astype(str) + ":00",
                                         format="%m/%d/%Y %H:%M", errors="coerce")
        dfs.append(df[["datetime", "temp"]])

    combined = pd.concat(dfs)
    avg_temp = (
        combined.groupby("datetime")["temp"]
        .mean()
        .reset_index()
        .rename(columns={"temp": "avg_temp"})
    )
    return avg_temp.sort_values("datetime").reset_index(drop=True)


# ── Recency effect features ──────────────────────────────────────────────────

def add_recency_features(
    df: pd.DataFrame,
    temp_col: str = "avg_temp",
    lag_hours: list[int] | None = None,
    ma_days: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged hourly temperatures and daily moving average temperatures
    as described in Wang et al. (2016) and Hong (2010).

    Parameters
    ----------
    df        : DataFrame with a temp_col column, sorted by datetime
    temp_col  : name of the temperature column
    lag_hours : list of hourly lags to add (default: [24] — one day)
    ma_days   : list of day-lags for moving averages (default: [1] — one day avg)

    Returns
    -------
    DataFrame with new lag and moving-average columns
    """
    if lag_hours is None:
        lag_hours = [24]
    if ma_days is None:
        ma_days = [1]

    df = df.copy()

    # Hourly lags: T_{t-h}
    for h in lag_hours:
        df[f"{temp_col}_lag{h}h"] = df[temp_col].shift(h)

    # Daily moving averages: T̃_{t,d} = mean of hours [24d-23 .. 24d]
    for d in ma_days:
        window = 24 * d
        df[f"{temp_col}_ma{d}d"] = df[temp_col].shift(1).rolling(window=window).mean()

    return df.dropna()


# ── Shifted-date scenario generation ─────────────────────────────────────────

def generate_shifted_scenarios(
    weather_df: pd.DataFrame,
    forecast_year: int,
    n_shift: int = 2,
    temp_col: str = "avg_temp",
) -> pd.DataFrame:
    """
    Generate temperature scenarios using the shifted-date method (Xie, 2016).

    For each historical year, shift the temperature series ±n_shift days
    to produce (2*n_shift + 1) scenarios per year.

    Parameters
    ----------
    weather_df    : DataFrame with 'datetime' and temp_col columns
    forecast_year : year to forecast (target year)
    n_shift       : number of days to shift in each direction (default: 2)
    temp_col      : name of the temperature column

    Returns
    -------
    DataFrame where each column is one scenario: 'scenario_<year>_shift<d>'
    Index is hour-of-year (0..8759)
    """
    df = weather_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year

    # Get all historical years excluding forecast year
    hist_years = sorted(df["year"].unique())
    hist_years = [y for y in hist_years if y != forecast_year]

    # Forecast year calendar (for alignment)
    target_hours = df[df["year"] == forecast_year][temp_col].values
    n_hours = len(target_hours)

    scenarios = {}
    for year in hist_years:
        year_temps = df[df["year"] == year][temp_col].values
        # Pad / trim to same length as target year
        for shift in range(-n_shift, n_shift + 1):
            shifted = np.roll(year_temps, shift * 24)
            # Trim or pad to n_hours
            if len(shifted) >= n_hours:
                scenarios[f"scenario_{year}_shift{shift:+d}"] = shifted[:n_hours]
            else:
                pad = np.full(n_hours - len(shifted), np.nan)
                scenarios[f"scenario_{year}_shift{shift:+d}"] = np.concatenate([shifted, pad])

    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df["actual"] = target_hours[:n_hours]
    return scenarios_df
