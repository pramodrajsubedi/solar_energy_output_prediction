# Solar Energy Output Prediction

A three-phase machine learning pipeline for predicting solar energy output from meteorological and solar geometry features, progressively expanding from a 6-feature linear baseline to a 24-feature multi-model ensemble trained on NSRDB data.


---

## Pipeline Overview

| Notebook | Phase | Goal |
|---|---|---|
| `phase_1.ipynb` | Baseline | 6-feature Linear Regression on Power Output |
| `phase_2.ipynb` | Expanded features | 12-feature Linear Regression with weather variables |
| `ml.ipynb` | Full ensemble | 24-feature multi-model comparison on GHI prediction |

---

## Phase 1 — Baseline Model

**Target:** Solar panel Power Output (kW)

**Target formula:**
```
Power_Output = GHI × Panel_Area × Panel_Efficiency × Temp_Loss_Factor / 1000
```
Where `Panel_Area = 1000 m²`, `Panel_Efficiency = 18%`, `Temp_Coefficient = -0.4%/°C`

**Features (6):**

| Feature | Role |
|---|---|
| GHI | Primary solar irradiance predictor |
| DNI | Direct normal irradiance |
| Temperature | Panel efficiency via temperature coefficient |
| Solar Zenith Angle | Sun position |
| Day_of_Year | Seasonal pattern |
| Hour_of_Day | Diurnal pattern |

**Model:** Linear Regression with StandardScaler  
**Split:** 80% train / 20% test, random_state=42

---

## Phase 2 — Expanded Features

**Target:** Same Power Output formula as Phase 1

**Additional features over Phase 1 (12 total):**

| Added Feature | Role |
|---|---|
| DHI | Diffuse horizontal irradiance |
| Relative Humidity | Weather condition |
| Wind Speed | Cooling effect on panels |
| Wind Direction | Weather pattern |
| Pressure | Atmospheric condition |
| Dew Point | Moisture content |

**Model:** Linear Regression  
**Split:** 80% train / 20% test, shuffle=False (preserves time order)

---

## ml.ipynb — Full Multi-Model Ensemble

**Target:** GHI (W/m²) — direct irradiance prediction

**Data:** Multiple NSRDB CSV files combined from `./all_solar_data/`, filtered for `Fill Flag == 0`

**Feature Engineering (24 features):**

| Category | Features |
|---|---|
| Time | Hour, Month, DayOfYear |
| Categorical time | Is_morning, Is_afternoon, is_summer, is_winter |
| Solar geometry | Solar Zenith Angle, cos_zenith_angle |
| Meteorological | Temperature, Dew Point, Relative Humidity, Pressure, Wind Speed, Wind Direction, Precipitable Water |
| Clearsky | Clearsky GHI, Clearsky DHI, Clearsky DNI |
| Interaction | temp_humidity_interaction, temp_difference |
| Lagged | GHI_prev_hour, GHI_prev_3hours |
| Rolling | GHI_rolling_mean_6h |

**Models Compared:**

| Model | Configuration |
|---|---|
| Linear Regression | Baseline |
| Random Forest | n_estimators=100, max_depth=15 |
| Gradient Boosting | n_estimators=100, lr=0.1, max_depth=7 |

**Best model:** Gradient Boosting Regressor (lowest Test RMSE, highest Test R²)

---

## Critical Finding — Data Leakage Identified

> In Phase 1 and Phase 2, the target variable `Power_Output` is derived directly from `GHI` using a known physics formula, while `GHI` is also included as a feature. This creates a deterministic relationship between a feature and the target — a form of data leakage that inflates R² scores. This was identified and documented as part of the learning process.

In `ml.ipynb`, the target shifts to raw `GHI` prediction, which represents a cleaner formulation — though `Clearsky GHI` and lagged GHI features still require careful interpretation.

---

## Architecture

```
NSRDB Data (multiple CSV files)
        │
        ▼
ml.ipynb                    ← Data loading, cleaning (Fill Flag == 0 filter)
        │
   ┌────┴────────────────┐
   ▼                     ▼
phase_1.ipynb         phase_2.ipynb
6-feature baseline    12-feature expanded
Linear Regression     Linear Regression
        │                     │
        └────────┬────────────┘
                 ▼
            ml.ipynb
    24-feature multi-model ensemble
    LR vs Random Forest vs Gradient Boosting
    Feature importance (Gradient Boosting)
    Model comparison: RMSE + R²
```

---

## Visualizations Produced

- Predictions vs Actual scatter plot
- Residual plot and residual distribution
- Feature importance (coefficients for LR, feature_importances_ for GB)
- Model comparison bar charts (RMSE and R²)
- Time series overlay: Actual vs Predicted (first 500 points)
- Overfitting analysis (Train R² vs Test R² gap)

---

## Stack

| Tool | Role |
|---|---|
| Python 3.10+ | Core language |
| pandas · numpy | Data processing and feature engineering |
| scikit-learn | Linear Regression, Random Forest, Gradient Boosting, StandardScaler |
| Matplotlib · Seaborn | Visualization |
| NSRDB | Solar irradiance data source |
| Jupyter Notebooks | Phased development |

---

## Author

**Pramod Raj Subedi**
[LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/)
