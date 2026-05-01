"""
probabilistic_forecasting.py
-----------------------------
Probabilistic Load Forecasting using the shifted-date temperature scenario method.
Mirrors: temp_scenarios_shifteddate(ProbLF)-2023.ipynb

Method (Xie, 2016):
  - Shift historical temperature series ±n days to generate (2n+1) × k scenarios
  - For each scenario: run the Vanilla + Recency forecast → get load scenario
  - Visualize as fan charts: monthly energy (kWh sum) and monthly peak (kW max)

With 7 years of weather (2018–2024) and n=2 day shift:
  Scenarios per year = 5, historical years = 5 (exclude 2023 forecast year)
  Total = 25 scenarios

Usage
-----
    from probabilistic_forecasting import run_plf
    results = run_plf(df_load, df_weather, forecast_year=2023, n_shift=2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vanilla_benchmark import VanillaBenchmarkModel


# ── Core PLF pipeline ─────────────────────────────────────────────────────────

def run_plf(
    df_load: pd.DataFrame,
    df_weather: pd.DataFrame,
    forecast_year: int = 2023,
    n_shift: int = 2,
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    load_col: str = "value",
    temp_col: str = "avg_temp",
    use_recency: bool = True,
    use_holiday: bool = True,
    group_label: str = "Heat Pump",
    save_path: str | None = None,
) -> dict:
    """
    Run full PLF pipeline and generate fan charts.

    Parameters
    ----------
    df_load      : cleaned aggregate load DataFrame (sum across meters, hourly)
    df_weather   : weather DataFrame with 'datetime' and avg_temp columns
    forecast_year: year to forecast (2023 in the study)
    n_shift      : days to shift ±(default: 2 → 25 scenarios)
    group_label  : label for plot title (e.g. 'Heat Pump', 'Electricity', 'Gas')
    save_path    : if provided, save fan chart here

    Returns
    -------
    dict with keys: scenarios_monthly_energy, scenarios_monthly_peak, actuals
    """
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    df_load[timestamp_col] = pd.to_datetime(df_load[timestamp_col])

    # Training data: all years except forecast_year
    train_df = df_load[df_load[timestamp_col].dt.year != forecast_year].copy()
    actual_df = df_load[df_load[timestamp_col].dt.year == forecast_year].copy()

    # Fit model on training data
    model = VanillaBenchmarkModel(use_recency=use_recency, use_holiday=use_holiday)
    model.fit(train_df)

    # Generate temperature scenarios
    hist_years = [y for y in df_weather["datetime"].dt.year.unique() if y != forecast_year]
    forecast_temps = df_weather[df_weather["datetime"].dt.year == forecast_year][temp_col].values

    n_hours = len(forecast_temps)
    forecast_dates = df_weather[df_weather["datetime"].dt.year == forecast_year]["datetime"].values

    scenarios = {}
    for year in hist_years:
        year_temps = df_weather[df_weather["datetime"].dt.year == year][temp_col].values
        for shift in range(-n_shift, n_shift + 1):
            key = f"{year}_shift{shift:+d}"
            shifted = np.roll(year_temps, shift * 24)
            if len(shifted) >= n_hours:
                scenarios[key] = shifted[:n_hours]
            else:
                scenarios[key] = np.concatenate([shifted, np.full(n_hours - len(shifted), np.nan)])

    print(f"Generated {len(scenarios)} temperature scenarios")

    # Forecast for each scenario
    monthly_energy = {}
    monthly_peak   = {}

    for key, temps in scenarios.items():
        scen_df = pd.DataFrame({
            timestamp_col: forecast_dates[:n_hours],
            temp_col:      temps,
        }).dropna()

        # Merge with load structure for calendar variables
        scen_df = scen_df.merge(actual_df[[timestamp_col]].assign(dummy=1).drop("dummy", axis=1),
                                 on=timestamp_col, how="inner")
        scen_df[load_col] = 0  # placeholder — we only need temperature + calendar

        try:
            preds = model.predict(scen_df)
            scen_df["forecast"] = preds
            scen_df["month"] = pd.to_datetime(scen_df[timestamp_col]).dt.month

            monthly_energy[key] = scen_df.groupby("month")["forecast"].sum().values
            monthly_peak[key]   = scen_df.groupby("month")["forecast"].max().values
        except Exception as e:
            print(f"  Skipped scenario {key}: {e}")

    # Actual monthly aggregates
    actual_df["month"] = actual_df[timestamp_col].dt.month
    actual_energy = actual_df.groupby("month")[load_col].sum().values
    actual_peak   = actual_df.groupby("month")[load_col].max().values

    # Plot fan charts
    _plot_fan_charts(
        monthly_energy=monthly_energy,
        monthly_peak=monthly_peak,
        actual_energy=actual_energy,
        actual_peak=actual_peak,
        group_label=group_label,
        forecast_year=forecast_year,
        n_shift=n_shift,
        save_path=save_path,
    )

    return {
        "scenarios_monthly_energy": monthly_energy,
        "scenarios_monthly_peak":   monthly_peak,
        "actual_energy":            actual_energy,
        "actual_peak":              actual_peak,
    }


def _plot_fan_charts(
    monthly_energy: dict,
    monthly_peak: dict,
    actual_energy: np.ndarray,
    actual_peak: np.ndarray,
    group_label: str,
    forecast_year: int,
    n_shift: int,
    save_path: str | None = None,
):
    """
    Plot side-by-side fan charts for monthly energy and monthly peak load.
    Reproduces Figures 4-12 through 4-14 from the capstone report.
    """
    months = np.arange(1, 13)
    n_scenarios = len(monthly_energy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Monthly Energy & Peak — Actual (dots) vs {n_scenarios} Forecast Scenarios "
        f"(shifted_date-{n_shift}day) — {group_label.upper()}",
        fontsize=11
    )

    for key in monthly_energy:
        axes[0].plot(months, monthly_energy[key], color="gray", alpha=0.5, linewidth=0.8)
        axes[1].plot(months, monthly_peak[key],   color="gray", alpha=0.5, linewidth=0.8)

    axes[0].plot(months, actual_energy, "ko", markersize=5, zorder=5, label="Actual")
    axes[1].plot(months, actual_peak,   "ko", markersize=5, zorder=5, label="Actual")

    for ax, ylabel, title in zip(
        axes,
        ["Energy kWh (Sum of Load)", "Peak Load kW"],
        [f"Monthly Energy — {forecast_year}", f"Monthly Peak — {forecast_year}"],
    ):
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(months)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Fan chart saved to {save_path}")
    plt.show()
