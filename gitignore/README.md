# Heat Pump Load Analysis and Forecasting

> **Author:** Pavani Yelisetti — M.S. Engineering Management (Energy Analytics Concentration)  
> **Supervisor:** Dr. Tao Hong — BigDEAL Lab, UNC Charlotte  
> **Capstone:** Master of Science in Engineering Management, 2026

---

## Overview

This research characterizes residential customers by their heating system type using smart meter electricity consumption data from Brunswick County, NC. Starting from raw AMI (Advanced Metering Infrastructure) data — which contains missing values, anomalies, and unclassified households — the project delivers:

1. A structured **data cleansing pipeline** for smart meter data
2. A **regression-based load forecasting model** (Vanilla Benchmark + Recency + Holiday effects)
3. A **customer classification framework** that identifies heat pump, electric, and gas heating households purely from load behavior
4. **Probabilistic Load Forecasting (PLF)** using the shifted-date temperature scenario method

**Key finding:** Heat pump systems consume approximately **0.1 kWh/°F less** heating energy than conventional electric resistance systems — a quantified basis for utility incentive program design.

---

## Data

| Dataset | Description |
|---|---|
| **Labeled (heat pump)** | 199 households with known heat pump use — 6M hourly records (2021–2024) |
| **Unlabeled** | ~21,000 households, ~700M records — ~90% expected heat pumps |
| **Weather** | 7 years (2018–2024) hourly data from 4 meteorological stations in Brunswick County |
| **Weather variables** | Temperature, humidity, heat index, pressure, visibility, wind speed, wind chill |
| **Granularity** | Hourly load in kW, averaged temperature across 4 stations |

Weather stations are combined using simple averaging:

$$T_t = \frac{1}{4} \sum_{i=1}^{4} T_{i,t}$$

---

## Pipeline

```
Raw Smart Meter Data (199 labeled + 21,000 unlabeled)
         │
         ▼
┌─────────────────────┐
│  1. Data Cleansing  │  Remove bad meters (heatmap), handle natural events,
│                     │  detect abnormal behaviors, iterative outlier removal
└────────┬────────────┘
         │  120 reliable labeled meters remain
         ▼
┌─────────────────────┐
│  2. Load Forecasting│  Vanilla Benchmark + Recency + Holiday effects
│     (MLR Model)     │  295 variables (Vanilla) → 495 with recency
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Classification  │  Piecewise regression + MAE comparison
│                     │  → Heat pump (98) / Electricity (76) / Gas (15)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. PLF (Shifted    │  25 temperature scenarios via ±2-day shift
│     Date Method)    │  Monthly energy & peak load fan charts
└─────────────────────┘
```

---

## Methods

### Tao's Vanilla Benchmark Model

$$\text{Load}_t = \beta_0 + \beta_1 \text{trend}_t + \beta_2 M_t + \beta_3 W_t + \beta_4 H_t + \beta_5 H_t W_t + f(T_t)$$

$$f(T_t) = \beta_6 T_t + \beta_7 T_t^2 + \beta_8 T_t^3 + \beta_9 T_t M_t + \beta_{10} T_t^2 M_t + \beta_{11} T_t^3 M_t + \beta_{12} T_t H_t + \beta_{13} T_t^2 H_t + \beta_{14} T_t^3 H_t$$

- **M** = Month (12 classes), **W** = Weekday (7 classes), **H** = Hour (24 classes)
- Total: **295 variables**

### Recency Effect Extension

Adds one-day lag temperature and one-day moving average temperature:

$$\text{Load}_t = \text{Vanilla} + \sum_d f(\tilde{T}_{t,d}) + \sum_h f(T_{t-h})$$

- Total: **495 variables**

### Holiday Effect

Major holidays (Christmas, Thanksgiving, etc.) are assigned a distinct weekday class to prevent their genuine high-load or travel-related patterns from being flagged as outliers.

### Outlier Detection (Iterative APE)

- Fit model in-sample on all data → compute APE per observation
- Remove top N% of APE timestamps (100% → 80% → 60% → 50% → 40%)
- Repeat until convergence

| Model variant | Total outliers identified |
|---|---|
| Vanilla + 1-day lag temperature | 560 |
| Vanilla + 1-day moving average | 476 |
| Vanilla + lag + moving avg + holiday | **473** (best — preserves holiday readings) |

### Customer Classification

- Train model on 98 reliable heat pump meters
- Apply trained model to each unlabeled household
- Compute normalized MAE (MAE / avg\_load)
- High normalized MAE → behavior deviates from heat pump pattern
- Cross-classify using piecewise load–temperature slope analysis

**Results:**

| Class | Meters | Signature |
|---|---|---|
| Heat pump | 98 | Balanced U-shape: moderate both seasons |
| Electricity | 76 | Sharp winter spikes, lower summer |
| Gas | 15 | Low/flat winter electricity, higher summer |

### Customer Binning (Slope-Based)

Heat pump meters binned by load–temperature slope on the heating side (<35°F):

| Bin | Slope range | Meters |
|---|---|---|
| Bin 1 | 0 to –0.049 | 12 |
| Bin 2 | –0.05 to –0.099 | 45 |
| Bin 3 | –0.10 to –0.149 | 31 |
| Bin 4 | –0.15 and below | 10 |

Bins 2+3 (76 meters) represent typical heat pump behavior and are used for cross-group comparison.

### Probabilistic Load Forecasting (Shifted-Date Method)

- 7 years weather data + 4 years load data available
- Shift historical temperature ±n days to generate (2n+1) × k scenarios
- 2-day shift → **25 scenarios** for 2023 forecast year
- Output: monthly energy (kWh sum) and monthly peak (kW max) fan charts

---

## Repository Structure

```
heat-pump-load-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── .github/workflows/ci.yml
│
├── src/
│   ├── weather.py                        # Station averaging + lag/MA temperature features
│   ├── cleansing.py                      # Heatmap bad-meter detection, natural event flags,
│   │                                     # abnormal behavior removal
│   ├── outlier_detection.py              # Iterative APE-based outlier removal
│   │                                     # (mirrors insample_outlierdetect-recency+holiday.ipynb)
│   ├── vanilla_benchmark.py              # Tao's Vanilla Benchmark MLR (295 vars)
│   ├── recency_model.py                  # Vanilla + recency effect (495 vars)
│   ├── classification.py                 # Piecewise regression, MAE-based customer segmentation
│   │                                     # (mirrors Electri_vs_bin_plots.ipynb)
│   ├── slope_binning.py                  # Slope-based heat pump binning
│   │                                     # (mirrors slope_120_meters-aggregate.ipynb)
│   └── probabilistic_forecasting.py      # Shifted-date PLF
│                                         # (mirrors temp_scenarios_shifteddate_ProbLF_-2023.ipynb)
│
├── notebooks/                            # Drop your .ipynb files here
│   ├── insample_outlierdetect-recency+holiday.ipynb
│   ├── slope_120_meters-aggregate.ipynb
│   ├── temp_scenarios_shifteddate_ProbLF_-2023.ipynb
│   └── Electri_vs_bin_plots.ipynb
│
├── data/                                 # Not committed — add your own
│   ├── weather/                          # wthrdump_*.csv from 4 stations
│   └── smart_meters/                     # Labeled (199) and unlabeled (21K) datasets
│
├── results/
│   └── figures/                          # Generated plots
│
└── docs/
    ├── capstone_report.pdf
    └── presentation.pptx
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/heat-pump-load-analysis.git
cd heat-pump-load-analysis
pip install -r requirements.txt

# 2. Place data
#    weather CSVs → data/weather/
#    smart meter CSVs → data/smart_meters/

# 3. Run full pipeline
python src/cleansing.py          # Bad meter removal → cleaned dataset
python src/outlier_detection.py  # Iterative APE outlier removal
python src/classification.py     # Customer segmentation
python src/probabilistic_forecasting.py  # PLF fan charts

# Or run notebooks interactively
jupyter notebook notebooks/
```

---

## Key Results

- **Heat pump vs. electric heating:** ~0.1 kWh/°F energy savings in heating season
- **Classification accuracy:** Clear separation of 3 customer classes via load–temperature slope
- **PLF:** 25 scenarios closely track actual 2023 monthly energy and peak load across all three customer classes
- **Outlier detection:** Best model (Vanilla + lag1 + moving avg + holiday) identified 473 true outliers, correctly preserving holiday-period readings

---

## Implications for Utilities

- Identify customer heating/cooling technology directly from smart meter data — no survey required
- Heat pump adoption reduces peak demand (more valuable than energy reduction alone)
- Enables design of targeted incentive programs: time-of-use rates, rebates, rider mechanisms
- Methodology is transferable to other regions, cooperatives, and climate zones

---

## References

1. Wang, Y.; Chen, Q.; Hong, T.; Kang, C. Review of Smart Meter Data Analytics. *IEEE Trans. Smart Grid*, 2018.
2. Hong, T. Short Term Electric Load Forecasting. Dissertation, 2010.
3. Hong, T.; Pinson, P.; Fan, S. Global Energy Forecasting Competition 2012. *Int. J. Forecast.*, 2014.
4. Wang, P.; Liu, B.; Hong, T. Electric load forecasting with recency effect. *Int. J. Forecast.*, 2016.
5. Hong, T.; Wang, P.; White, L. Weather station selection for electric load forecasting. *Int. J. Forecast.*, 2015.
6. Sobhani, M. et al. Combining weather stations for electric load forecasting. *Energies*, 2019.
7. Xie, J.; Hong, T. GEFCom2014 probabilistic electric load forecasting. *Int. J. Forecast.*, 2016.

---

## Citation

```bibtex
@mastersthesis{yelisetti2026heatpump,
  author  = {Pavani Yelisetti},
  title   = {Heat Pump Load Analysis and Forecasting},
  school  = {University of North Carolina at Charlotte},
  year    = {2026},
  advisor = {Dr. Tao Hong}
}
```
