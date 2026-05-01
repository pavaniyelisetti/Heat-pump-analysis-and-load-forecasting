"""
vanilla_benchmark.py
--------------------
Tao's Vanilla Benchmark Model for short-term electric load forecasting,
extended with Recency Effect and Holiday Effect.

References:
  - Hong (2010): Original Vanilla Benchmark (295 variables)
  - Wang, Liu, Hong (2016): Recency Effect extension (495 variables)
  - Xie (2016): Holiday effect, APE-based outlier detection

Model notation:
  Load_t = β0 + β1·trend_t + β2·Month_t + β3·Weekday_t + β4·Hour_t
           + β5·Hour_t·Weekday_t + f(T_t)
           [+ Σ_d f(T̃_{t,d}) + Σ_h f(T_{t-h})]   ← recency effect

  f(T_t) = β6·T + β7·T² + β8·T³
           + β9·T·Month + β10·T²·Month + β11·T³·Month
           + β12·T·Hour + β13·T²·Hour + β14·T³·Hour
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ── US Federal Holidays (major ones used in Holiday Effect) ──────────────────
MAJOR_HOLIDAYS = {
    "new_year":        lambda year: pd.Timestamp(f"{year}-01-01"),
    "memorial_day":    lambda year: _nth_weekday(year, 5, 0, -1),   # last Mon May
    "independence":    lambda year: pd.Timestamp(f"{year}-07-04"),
    "labor_day":       lambda year: _nth_weekday(year, 9, 0, 1),    # 1st Mon Sep
    "thanksgiving":    lambda year: _nth_weekday(year, 11, 3, 4),   # 4th Thu Nov
    "christmas":       lambda year: pd.Timestamp(f"{year}-12-25"),
}

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return the nth occurrence (1-based, or -1 for last) of a weekday in a month."""
    import calendar
    first = pd.Timestamp(f"{year}-{month:02d}-01")
    days = pd.date_range(first, periods=calendar.monthrange(year, month)[1], freq="D")
    matches = days[days.dayofweek == weekday]
    return matches[n - 1] if n > 0 else matches[n]


def get_holiday_dates(years: list[int]) -> set:
    """Return set of holiday dates (as date objects) for all given years."""
    dates = set()
    for year in years:
        for _, fn in MAJOR_HOLIDAYS.items():
            try:
                dates.add(fn(year).date())
            except Exception:
                pass
    return dates


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    temp_col: str = "avg_temp",
    load_col: str = "value",
    datetime_col: str = "timestamp",
    use_recency: bool = True,
    use_holiday: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix for the Vanilla + Recency + Holiday model.

    Parameters
    ----------
    df           : DataFrame with timestamp, load, and averaged temperature
    use_recency  : include 1-day lag and 1-day moving average temperature terms
    use_holiday  : encode major holidays as a separate weekday category (= 7)

    Returns
    -------
    X : feature DataFrame
    y : load series
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Calendar variables
    df["trend"]   = np.arange(len(df))
    df["month"]   = df[datetime_col].dt.month          # 1–12
    df["weekday"] = df[datetime_col].dt.dayofweek      # 0=Mon, 6=Sun
    df["hour"]    = df[datetime_col].dt.hour           # 0–23

    # Holiday effect: recode major holidays as weekday = 7
    if use_holiday:
        years = df[datetime_col].dt.year.unique().tolist()
        holidays = get_holiday_dates(years)
        df.loc[df[datetime_col].dt.date.isin(holidays), "weekday"] = 7

    T = df[temp_col]

    # Dummy encoding for categorical variables
    month_dummies   = pd.get_dummies(df["month"],   prefix="M",  drop_first=True)
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="WD", drop_first=True)
    hour_dummies    = pd.get_dummies(df["hour"],    prefix="H",  drop_first=True)

    # Temperature polynomials
    T2, T3 = T ** 2, T ** 3

    # Interaction terms: T × Month, T² × Month, T³ × Month
    T_month  = month_dummies.multiply(T,  axis=0).add_suffix("_T")
    T2_month = month_dummies.multiply(T2, axis=0).add_suffix("_T2")
    T3_month = month_dummies.multiply(T3, axis=0).add_suffix("_T3")

    # Interaction terms: T × Hour, T² × Hour, T³ × Hour
    T_hour  = hour_dummies.multiply(T,  axis=0).add_suffix("_T")
    T2_hour = hour_dummies.multiply(T2, axis=0).add_suffix("_T2")
    T3_hour = hour_dummies.multiply(T3, axis=0).add_suffix("_T3")

    # Hour × Weekday interaction
    HW = hour_dummies.multiply(df["weekday"], axis=0).add_suffix("_WD")

    parts = [
        df[["trend"]],
        month_dummies,
        weekday_dummies,
        hour_dummies,
        HW,
        T.rename("T"), T2.rename("T2"), T3.rename("T3"),
        T_month, T2_month, T3_month,
        T_hour, T2_hour, T3_hour,
    ]

    # Recency effect: 1-day lag and 1-day moving average
    if use_recency:
        T_lag1 = T.shift(24).rename("T_lag1d")
        T_ma1  = T.shift(1).rolling(24).mean().rename("T_ma1d")
        T_lag1_2  = T_lag1 ** 2
        T_lag1_3  = T_lag1 ** 3
        T_ma1_2   = T_ma1 ** 2
        T_ma1_3   = T_ma1 ** 3

        T_lag1_M  = month_dummies.multiply(T_lag1,  axis=0).add_suffix("_lag1_T")
        T_lag1_H  = hour_dummies.multiply(T_lag1,   axis=0).add_suffix("_lag1_T")
        T_ma1_M   = month_dummies.multiply(T_ma1,   axis=0).add_suffix("_ma1_T")
        T_ma1_H   = hour_dummies.multiply(T_ma1,    axis=0).add_suffix("_ma1_T")

        parts += [
            T_lag1, T_lag1_2.rename("T_lag1d_2"), T_lag1_3.rename("T_lag1d_3"),
            T_ma1,  T_ma1_2.rename("T_ma1d_2"),  T_ma1_3.rename("T_ma1d_3"),
            T_lag1_M, T_lag1_H, T_ma1_M, T_ma1_H,
        ]

    X = pd.concat(parts, axis=1).dropna()
    y = df.loc[X.index, load_col]

    return X, y


# ── Model ────────────────────────────────────────────────────────────────────

class VanillaBenchmarkModel:
    """
    Tao's Vanilla Benchmark MLR with optional Recency and Holiday extensions.

    Usage
    -----
    model = VanillaBenchmarkModel(use_recency=True, use_holiday=True)
    model.fit(train_df)
    preds = model.predict(test_df)
    ape   = model.ape(test_df)        # for outlier detection
    """

    def __init__(self, use_recency: bool = True, use_holiday: bool = True):
        self.use_recency = use_recency
        self.use_holiday = use_holiday
        self._model = LinearRegression()
        self._feature_cols = None

    def fit(self, df: pd.DataFrame, **kwargs) -> "VanillaBenchmarkModel":
        X, y = build_features(df, use_recency=self.use_recency,
                               use_holiday=self.use_holiday, **kwargs)
        self._feature_cols = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        X, _ = build_features(df, use_recency=self.use_recency,
                               use_holiday=self.use_holiday, **kwargs)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        return self._model.predict(X)

    def ape(self, df: pd.DataFrame, load_col: str = "value", **kwargs) -> pd.Series:
        """Compute Absolute Percentage Error for each observation."""
        X, y = build_features(df, use_recency=self.use_recency,
                               use_holiday=self.use_holiday, **kwargs)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        pred = pd.Series(self._model.predict(X), index=y.index)
        ape = (np.abs(pred - y) / y.abs()).replace([np.inf, -np.inf], np.nan) * 100
        return ape

    def mape(self, df: pd.DataFrame, **kwargs) -> float:
        X, y = build_features(df, use_recency=self.use_recency,
                               use_holiday=self.use_holiday, **kwargs)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        pred = self._model.predict(X)
        return mean_absolute_percentage_error(y, pred) * 100

    def mae(self, df: pd.DataFrame, **kwargs) -> float:
        X, y = build_features(df, use_recency=self.use_recency,
                               use_holiday=self.use_holiday, **kwargs)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        pred = self._model.predict(X)
        return mean_absolute_error(y, pred)
