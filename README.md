# Temperature, Air Pollution, and Daily Mortality: Distributed Lag Non-linear Models (DLNM) & Case-Crossover Evaluation in Chicago (1987–2000)

[![R Version](https://img.shields.io/badge/R-%3E%3D%204.1.0-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Field](https://img.shields.io/badge/Field-Environmental%20Epidemiology-green.svg)]()

---

## 📌 Background & Research Questions

Exposure to ambient temperature extremes and ambient particulate matter ($PM_{10}$) poses significant acute and delayed burdens on public health. Investigating these risks requires rigorous epidemiological designs capable of capturing:
1. **Non-linear exposure-response shapes** (e.g., U/V-shaped relationships).
2. **Delayed and harvesting effects** across multi-week lag intervals.
3. **Control of long-term trends, seasonality, and time-invariant confounders**.

This project implements an end-to-end environmental epidemiology pipeline comparing **aggregate time-series models (DLNM in quasi-Poisson GLMs)** with **individual-level time-stratified case-crossover designs (conditional quasi-Poisson via GNM)**. 

### Core Research Questions
* What is the Minimum Mortality Temperature (MMT), and what are the cumulative risks of extreme cold (1st percentile) and extreme heat (99th percentile) on daily mortality?
* What are the latency patterns (lag structure) differentiating cold and heat effects?
* Does short-term particulate matter ($PM_{10}$, lag 0–1) exposure independently increase all-cause and cardiovascular (CVD) mortality after adjusting for non-linear temperature and co-pollutants ($O_3$)?
* How consistent are the estimates between standard time-series regression and time-stratified case-crossover designs?

---

## 🗂 Data & Structure

The analysis is based on the **Chicago NMMAPS (National Morbidity, Mortality, and Air Pollution Study)** daily time-series data (1987–2000, $N = 5,114$ days) available in the `dlnm` R package.

### Key Variables
* **Outcomes**: `death` (Total non-accidental mortality), `cvd` (Cardiovascular deaths), `resp` (Respiratory deaths).
* **Exposures**: `temp` (Mean daily temperature, °C), `pm10` (Particulate matter $\le 10\,\mu\text{m}$, $\mu\text{g/m}^3$), `o3` (Ozone, ppb).
* **Meteorological Confounders**: `dptp` (Dew point temperature), `rhum` (Relative humidity).
* **Temporal Indicators**: `date`, `dow` (Day of week), `time` ($1 \dots N$).

### Repository Layout
```text
.
├── R/
│   └── 01_dlnm_case_crossover_pipeline.R  # Complete reproducible pipeline
├── figures/
│   ├── cumulative_exposure_response.png   # Fig 1: Overall cumulative curve
│   ├── lag_response_slices.png            # Fig 2: Cold vs Heat lag curves
│   └── exposure_lag_contour.png           # Fig 3: 2D contour surface
├── README.md                              # Scientific & methodological report
└── LICENSE
