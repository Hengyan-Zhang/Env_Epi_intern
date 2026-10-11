# ==============================================================================
# Environmental Epidemiology Analysis Pipeline:
# Time-Series (DLNM) & Time-Stratified Case-Crossover Designs
# Dataset: Chicago NMMAPS (1987–2000)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Package Installation and Loading
# ------------------------------------------------------------------------------

# Define list of required packages for this analysis
required_packages <- c(
  "tidyverse", # Data manipulation (dplyr, tidyr) and visualization (ggplot2)
  "dlnm",      # Distributed Lag Non-linear Models
  "splines",   # Basis spline regression functions (ns, bs)
  "gnm",       # Generalized Nonlinear Models (Conditional Poisson regression)
  "naniar",    # Missing data inspection and visualization
  "DBI",       # Generic database interface for R
  "RSQLite",   # SQLite database interface driver
  "sqldf"      # SQL execution wrapper on R data frames
)

# Identify packages that are not yet installed on this system
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

# Install any missing dependencies automatically
if (length(new_packages) > 0) {
  install.packages(new_packages)
}

# Attach packages to the current session
library(tidyverse)
library(dlnm)
library(splines)
library(gnm)
library(naniar)
library(DBI)
library(RSQLite)
library(sqldf)


# ------------------------------------------------------------------------------
# 2. Data Ingestion and Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------

# Load the built-in Chicago NMMAPS time-series dataset from dlnm
data(chicagoNMMAPS)

# Inspect internal structure and data types of each column
str(chicagoNMMAPS)

# Display summary statistics (mean, quartiles, min/max) for all columns
summary(chicagoNMMAPS)

# Check if any missing values (NA) exist in the entire dataset
anyNA(chicagoNMMAPS)

# Calculate total count of NA values across the entire dataset
sum(is.na(chicagoNMMAPS))

# Count missing values per individual row
rowSums(is.na(chicagoNMMAPS))

# Calculate proportion of missing values per column
colMeans(is.na(chicagoNMMAPS))

# Filter and view rows that contain at least one missing value
chicagoNMMAPS[!complete.cases(chicagoNMMAPS), ]

# Generate a detailed summary table of missing values by variable
miss_var_summary(chicagoNMMAPS)

# Plot missingness patterns across rows and columns
vis_miss(chicagoNMMAPS)

# Filter incomplete records using tidyverse pipeline syntax
chicagoNMMAPS |> filter(!if_all(everything(), complete.cases))

# Count total missing values for each column using across()
chicagoNMMAPS |> summarise(across(everything(), ~ sum(is.na(.))))

# Display variable column names in the dataset
names(chicagoNMMAPS)


# ------------------------------------------------------------------------------
# 3. Descriptive Summary Table & Preliminary Visualizations
# ------------------------------------------------------------------------------

# Compute distribution statistics (mean, IQR, missingness) for primary variables
summary_table <- chicagoNMMAPS |>
  summarise(across(
    c(death, temp, pm10),
    list(
      mean = ~ mean(., na.rm = TRUE),
      p25  = ~ quantile(., 0.25, na.rm = TRUE),
      p50  = ~ median(., na.rm = TRUE),
      p75  = ~ quantile(., 0.75, na.rm = TRUE),
      na_n = ~ sum(is.na(.))
    )
  )) |>
  pivot_longer(
    cols = everything(),
    names_to = c("variable", ".value"),
    names_pattern = "(.*)_(.*)"
  )

# Output summary metrics table to console
print(summary_table)

# Reshape key variables into long format for multi-panel time-series visualization
chicago_long <- chicagoNMMAPS |>
  mutate(date = as.Date(date)) |>
  select(date, death, temp, pm10) |>
  pivot_longer(
    cols = c(death, temp, pm10),
    names_to = "variable",
    values_to = "value"
  )

# Plot longitudinal multi-panel time-series with GAM trend lines
ggplot(chicago_long, aes(x = date, y = value)) +
  geom_line(colour = "grey60", alpha = 0.6, linewidth = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), colour = "firebrick", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ variable, scales = "free_y", ncol = 1) +
  theme_minimal() +
  labs(
    x = "Date",
    y = "Value",
    title = "Time Series Trajectories of Mortality, Temperature, and PM10"
  )

# Add explicit time index for spline modeling
chicagoNMMAPS$date <- as.Date(chicagoNMMAPS$date)
chicagoNMMAPS$time <- 1:nrow(chicagoNMMAPS)

# Plot raw bivariate relationship between temperature and mortality with loess smooth
ggplot(chicagoNMMAPS, aes(x = temp, y = death)) +
  geom_point(colour = "grey60", alpha = 0.5) +
  geom_smooth(colour = "firebrick", linewidth = 0.8, se = FALSE) +
  theme_minimal() +
  labs(
    x = "Temperature (°C)",
    y = "Daily Number of Deaths",
    title = "Unadjusted Bivariate Association: Temperature vs Mortality"
  )


# ------------------------------------------------------------------------------
# 4. Distributed Lag Non-linear Model (DLNM): Temperature & Mortality
# ------------------------------------------------------------------------------

# Define 1st and 99th temperature percentiles for extreme exposure slices
p1  <- quantile(chicagoNMMAPS$temp, 0.01, na.rm = TRUE)
p99 <- quantile(chicagoNMMAPS$temp, 0.99, na.rm = TRUE)

# Construct bi-dimensional cross-basis matrix for temperature
# argvar: Natural cubic splines (knots at 10th, 75th, 90th percentiles)
# arglag: Natural cubic splines over 0-21 lag days with log-spaced knots
cb_temp <- crossbasis(
  chicagoNMMAPS$temp,
  lag = 21,
  argvar = list(
    fun   = "ns",
    knots = quantile(chicagoNMMAPS$temp, c(0.10, 0.75, 0.90), na.rm = TRUE)
  ),
  arglag = list(
    fun   = "ns",
    knots = logknots(21, 3)
  )
)

# Fit quasi-Poisson generalized linear model (GLM)
# ns(time, 7 * 14): 7 degrees of freedom per year across 14 years to control for trends and seasonality
# dow: Day-of-week factor to control for short-term activity cycles
model_dlnm <- glm(
  death ~ cb_temp + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = chicagoNMMAPS
)

# Preliminary prediction across temperature grid to determine reference point
pred_init <- crosspred(cb_temp, model_dlnm, by = 0.5)

# Locate Minimum Mortality Temperature (MMT) corresponding to lowest cumulative RR
mmt <- pred_init$predvar[which.min(pred_init$allRRfit)]
cat("Calculated Minimum Mortality Temperature (MMT):", round(mmt, 2), "°C\n")

# Re-run prediction using MMT as reference centering point, including exact slice targets
pred_grid <- sort(unique(c(seq(min(chicagoNMMAPS$temp), max(chicagoNMMAPS$temp), by = 0.5), p1, p99, mmt)))
pred_temp <- crosspred(cb_temp, model_dlnm, cen = mmt, at = pred_grid)

# Plot Fig 1 (Overall cumulative effect) and Fig 2 (Lag slices for P1 and P99) side-by-side
par(mfrow = c(1, 2))

# Plot overall cumulative exposure-response curve relative to MMT
plot(
  pred_temp, "overall",
  xlab = "Temperature (°C)",
  ylab = "Relative Risk (RR)",
  main = paste0("Overall Cumulative (Ref: ", round(mmt, 1), "°C)")
)
abline(h = 1, lty = 2, col = "grey50")

# Align exact percentiles with prediction grid
var_p1  <- pred_temp$predvar[which.min(abs(pred_temp$predvar - p1))]
var_p99 <- pred_temp$predvar[which.min(abs(pred_temp$predvar - p99))]

# Plot lag-specific effect curves for extreme cold (P1) and heat (P99)
plot(
  pred_temp, "slices", var = var_p1, col = "blue",
  ylab = "Relative Risk (RR)", xlab = "Lag (days)",
  main = paste0("Cold (", round(var_p1, 1), "°C) & Hot (", round(var_p99, 1), "°C)")
)
lines(pred_temp, "slices", var = var_p99, col = "red")
abline(h = 1, lty = 2, col = "grey50")
legend("topright", legend = c("P1 (Cold)", "P99 (Heat)"), col = c("blue", "red"), lty = 1, bty = "n")

# Reset graphic device parameters before calling filled.contour wrapper
par(mfrow = c(1, 1))

# Plot Fig 3: 2D contour surface of exposure-lag-response relationship
plot(
  pred_temp, "contour",
  xlab       = "Temperature (°C)",
  ylab       = "Lag (days)",
  key.title  = title("RR"),
  plot.title = title(main = "Exposure-Lag-Response Contour")
)

# Extract point estimates and 95% CI for extreme temperature scenarios relative to MMT
pred_extreme <- crosspred(cb_temp, model_dlnm, cen = mmt, at = c(p1, p99))

extreme_results <- tibble(
  Scenario = c("Extreme Cold (P1)", "Extreme Heat (P99)"),
  Temp_C   = c(p1, p99),
  RR       = round(pred_extreme$allRRfit, 4),
  CI_lower = round(pred_extreme$allRRlow, 4),
  CI_upper = round(pred_extreme$allRRhigh, 4)
)

print(extreme_results)


# ------------------------------------------------------------------------------
# 5. Multivariable Air Pollution Modeling & Sensitivity Analyses
# ------------------------------------------------------------------------------

# Derive cumulative exposures, seasonal stratifications, and matching strata
df_analysis <- chicagoNMMAPS %>%
  mutate(
    date       = as.Date(date),
    time       = 1:n(),
    dow        = as.factor(dow),
    year       = lubridate::year(date),
    month      = lubridate::month(date),
    # Compute moving average for PM10 over lag 0-1 days
    pm10_lag1  = dplyr::lag(pm10, 1),
    pm10_lag01 = (pm10 + pm10_lag1) / 2,
    # Compute moving average for O3 over lag 0-1 days
    o3_lag1    = dplyr::lag(o3, 1),
    o3_lag01   = (o3 + o3_lag1) / 2,
    # Define warm (May-Oct) and cold (Nov-Apr) season indicators
    season     = if_else(month %in% 5:10, "Warm", "Cold"),
    season     = as.factor(season),
    # Construct time stratum for case-crossover: Year x Month x Day-of-week
    stratum    = factor(paste(year, month, dow, sep = ":"))
  )

# Re-generate temperature cross-basis matrix aligned with derived dataset
cb_temp_aligned <- crossbasis(
  df_analysis$temp,
  lag = 21,
  argvar = list(fun = "ns", knots = quantile(df_analysis$temp, c(0.10, 0.75, 0.90), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(21, 3))
)

# Helper function: calculate RR and 95% CI per 10-unit increment of exposure
extract_rr_10 <- function(model, var_name, label = "Model") {
  coef_est <- coef(model)[var_name]
  se_est   <- sqrt(vcov(model)[var_name, var_name])
  
  rr   <- exp(coef_est * 10)
  low  <- exp((coef_est - 1.96 * se_est) * 10)
  high <- exp((coef_est + 1.96 * se_est) * 10)
  
  tibble(
    Model = label,
    RR    = round(rr, 4),
    CI_95 = sprintf("%.4f (%.4f, %.4f)", rr, low, high)
  )
}

# 5.1 Primary Single-Pollutant Model: PM10 (lag 0-1) adjusted for temperature DLNM
mod_base <- glm(
  death ~ pm10_lag01 + cb_temp_aligned + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_base <- extract_rr_10(mod_base, "pm10_lag01", "1. Primary (PM10 lag01)")

# 5.2 Dual-Pollutant Model: Co-adjusting for Ozone (O3 lag 0-1)
mod_bi <- glm(
  death ~ pm10_lag01 + o3_lag01 + cb_temp_aligned + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_bi <- extract_rr_10(mod_bi, "pm10_lag01", "2. Dual-pollutant (+ O3)")

# 5.3 Weather Confounding: Adjusting for Dew Point Temperature (dptp)
mod_dptp <- glm(
  death ~ pm10_lag01 + cb_temp_aligned + ns(dptp, df = 3) + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_dptp <- extract_rr_10(mod_dptp, "pm10_lag01", "3. Confounder (+ Dew Point)")

# 5.4 Effect Modification: Stratification by Season (Interaction Model)
mod_season <- glm(
  death ~ pm10_lag01:season + season + cb_temp_aligned + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_warm <- extract_rr_10(mod_season, "pm10_lag01:seasonWarm", "4a. Season: Warm")
res_cold <- extract_rr_10(mod_season, "pm10_lag01:seasonCold", "4b. Season: Cold")

# 5.5 Sensitivity Analysis: Varying Time Spline Degrees of Freedom (5 vs 10 df/year)
mod_df5 <- glm(
  death ~ pm10_lag01 + cb_temp_aligned + ns(time, df = 5 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_df5 <- extract_rr_10(mod_df5, "pm10_lag01", "5a. Sensitivity: df=5/yr")

mod_df10 <- glm(
  death ~ pm10_lag01 + cb_temp_aligned + ns(time, df = 10 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_df10 <- extract_rr_10(mod_df10, "pm10_lag01", "5b. Sensitivity: df=10/yr")

# Combine multivariable and sensitivity model results
summary_multivar <- bind_rows(res_base, res_bi, res_dptp, res_warm, res_cold, res_df5, res_df10)
print(summary_multivar)


# ------------------------------------------------------------------------------
# 6. Time-Stratified Case-Crossover Design vs. Time-Series Comparison
# ------------------------------------------------------------------------------

# 6.1 Conditional Poisson Regression for All-Cause Mortality
# eliminate = stratum absorbs matched cluster indicators (equivalent to conditional logistic regression)
mod_cc_all <- gnm(
  death ~ pm10_lag01 + cb_temp_aligned,
  eliminate = stratum,
  family    = quasipoisson(),
  data      = df_analysis
)
res_cc_all <- extract_rr_10(mod_cc_all, "pm10_lag01", "Case-Crossover (All Cause)")

# 6.2 Conditional Poisson Regression for Cardiovascular Mortality (cvd)
mod_cc_cvd <- gnm(
  cvd ~ pm10_lag01 + cb_temp_aligned,
  eliminate = stratum,
  family    = quasipoisson(),
  data      = df_analysis
)
res_cc_cvd <- extract_rr_10(mod_cc_cvd, "pm10_lag01", "Case-Crossover (CVD)")

# 6.3 Standard Time-Series Model for Cardiovascular Mortality (cvd) Comparison
mod_ts_cvd <- glm(
  cvd ~ pm10_lag01 + cb_temp_aligned + ns(time, df = 7 * 14) + dow,
  family = quasipoisson(),
  data   = df_analysis
)
res_ts_cvd <- extract_rr_10(mod_ts_cvd, "pm10_lag01", "Time-Series (CVD)")

# 6.4 Synthesis: Design Comparison Table
comparison_table <- bind_rows(
  tibble(Design = "Time-Series",    Outcome = "All-cause Death",      RR_95CI = res_base$CI_95),
  tibble(Design = "Case-Crossover", Outcome = "All-cause Death",      RR_95CI = res_cc_all$CI_95),
  tibble(Design = "Time-Series",    Outcome = "Cardiovascular (CVD)", RR_95CI = res_ts_cvd$CI_95),
  tibble(Design = "Case-Crossover", Outcome = "Cardiovascular (CVD)", RR_95CI = res_cc_cvd$CI_95)
)

print(comparison_table)