# SleepSense — Project Blueprint

## Project title
**SleepSense — India Edition**  
Predicting individual sleep quality (0–100) using lifestyle, digital and environmental data; packaged with a polished multilingual Streamlit app.

---

## High-level goals
- Build an interpretable supervised regression model to estimate a Sleep Quality Score (continuous 0–100).  
- Emphasize feature engineering that reflects Indian lifestyle and city-specific environment.  
- Demonstrate end-to-end ML workflow: dataset design, EDA, preprocessing, modeling (Linear / Ridge / Lasso), evaluation, interpretation, and deployment via Streamlit.  
- Produce a recruiter- and professor-ready deliverable (GitHub repo, demo app, project report, slides).

---

## Project scope & success criteria
**Scope**
- Dataset: 12,000 rows focused on Indian cities (urban/rural mix) with realistic distributions.  
- Models: Linear Regression (manual implementation using `sklearn` core where required), Ridge and Lasso (use `sklearn` where syllabus allows limited scikit-learn).  
- Delivery: Jupyter notebooks for each step, a polished Streamlit app, saved trained model(s), a reproducible data generator, README and presentation.

**Success criteria**
- Clear, reproducible pipeline (data → features → model → evaluation).  
- Interpretability: coefficient analysis, residual diagnostics, feature importance.  
- App: multilingual 30-level interpreter, interactive visualizations, downloadable report.

---

## Dataset: design & schema (real + synthetic approach)

**Principles**
- Realistic when possible: distributions mirror known demographic or environmental facts (age distribution, city populations, typical sleep hours).  
- Cultural attributes included: tea/coffee intake, spice intake, festival frequency, religious frequency.  
- City coverage: aim to include all major Indian cities and an "Other" bucket; ensure city-level variability (noise, pollution, metro indicator).

**Row count**: **12,000** records.

**Primary columns (features and types)**

| Column | Type | Description |
|---|---:|---|
| age | int | Age in years (8–80) |
| family_size | int | Number of family members |
| work_hours | float | Typical daily work hours |
| avg_sleep_hours | float | Average nightly sleep (hours) |
| screen_time_hours | float | Daily screen time (hours) |
| tea_cups | int | Tea cups per day |
| coffee_cups | int | Coffee cups per day |
| late_snack | int (0/1) | Late-night snack (binary) |
| spice_intake | int (1–5) | Self-reported spice level |
| religious_freq | int | Weekly religious activities (times/week) |
| festival_freq | int | Monthly festival participation (approx) |
| physical_activity_min | int | Minutes physical activity per day |
| bedtime_variability | float | Std dev or variability in bedtime (hrs) |
| stress_level | float | Self-reported 0–10 |
| city_noise_dB | float | Nighttime noise level (dB) |
| light_pollution_index | float | 0–100 light pollution proxy |
| temperature_night | float | Night temperature (°C) |
| humidity_night | float | Night humidity (%) |
| air_quality_index | float | AQI (0–500) |
| screen_brightness_behavior | int | Bright screen before bed (0/1) |
| sex_num | int | Encoded sex: 0,1,2 |
| is_metro | int | Metro city flag (0/1) |
| city | category | City name (Ahmedabad, Bengaluru, Chennai, Delhi, Mumbai, Pune, etc., + Other) |

**Engineered features (to compute)**
- sleep_deficit = max(0, 7 - avg_sleep_hours)  
- digital_fatigue = max(0, (screen_time_hours - 2) / 6)  
- env_stress = weighted sum of city_noise_dB, light_pollution_index, air_quality_index  
- lifestyle_balance = function(physical_activity_min, stress_level) scaled to [-1,1]  
- fatigue_env_interaction = digital_fatigue * env_stress  
- late_snack_effect = encoded effect of late_snack combined with spice_intake

**Target**
- sleep_quality_score ∈ [0, 100] (continuous).  
- Construction approach (synthetic + noise):
  - base_score = 70  
  - subtract weighted penalties: w1*env_stress + w2*stress_level + w3*sleep_deficit + w4*digital_fatigue + ...  
  - add positive weights for avg_sleep_hours, lifestyle_balance  
  - add city-specific bias for some metros to simulate unobserved factors  
  - clamp to 0..100 and add Gaussian noise ~ N(0, σ) to simulate measurement error.

**Rationale**: target built this way makes model interpretation straightforward (coefficient signs align with domain expectations), while still allowing meaningful residuals for diagnostics.

---

## Dataset generation plan (practical)
- Create per-feature realistic distributions:
  - age: mixture (students + working-age + older adults) — sample from truncated normal or custom bins.  
  - avg_sleep_hours: normal around 7 with tails; shift for shift-workers.  
  - screen_time_hours: skewed right (many low, some heavy users).  
  - city_noise_dB / AQI: city-wise grouped distributions (metros higher mean).  
- Stratify sampling across cities to ensure representation (e.g., 30–40 major cities plus Others).  
- Add metadata fields: sample_source = 'synthetic' and created_date.

---

## Data split & validation strategy
- Train / Validation / Test: **70% / 15% / 15%** (shuffle by rows).  
- Cross-validation: 5-fold CV for hyperparameter tuning (Ridge/Lasso alpha grid).  
- If city-level leakage concern exists, use grouped CV by city for robustness experiments.

---

## Preprocessing & feature engineering pipeline
1. Read raw CSV → inspect missing, basic stats.  
2. Impute missing numeric values (median) and categorical (mode). Record imputation strategy.  
3. Encode categorical features (city via one-hot, sex via integer).  
4. Scale numeric features as required (StandardScaler for linear models; note scaling must be saved for deployment).  
5. Create engineered features listed above.  
6. Save processed dataset to `data/processed/sleepsense_processed.csv`.

**Important note**: scaling used only after feature engineering. Keep scaler objects saved (`joblib`) for deployment.

---

## Modeling plan (aligns with faculty notebook)
- Notebook 04_model_building.ipynb:
  - Fit **Ordinary Least Squares (OLS)** linear regression (manual with `sklearn.linear_model.LinearRegression` but show normal equations in theoretical cell).  
  - Fit **Ridge** and **Lasso** using `sklearn.linear_model` for regularization demonstration (limited scikit-learn use per syllabus).  
- Notebook 05_model_evaluation.ipynb:
  - Metrics: MSE, RMSE, MAE, R², Adjusted R², MAPE.  
  - Residual analysis: histogram, QQ-plot, residual vs. fitted, heteroscedasticity check (Breusch-Pagan).  
  - Influence: Cook’s distance for outliers.  
- Notebook 06_model_tuning_and_regularization.ipynb:
  - Hyperparameter grid search for alpha (Ridge/Lasso). Use manual CV or `sklearn.model_selection.GridSearchCV` (brief, minimal use).  
- Notebook 07_final_interpretation.ipynb:
  - Coefficient table, direction & magnitude inferences, city-wise errors and fairness check.  
  - Save final model (joblib) and metrics to `models/`.

---

## Evaluation & interpretation
- Report primary metrics on test set (MSE, RMSE, MAE, R², Adjusted R²).  
- Summarize model coefficients in a clean table with short interpretations like the professor wants (direction, magnitude).  
- Residuals: check mean≈0 and reasonable variance. Present top 10 highest absolute errors with possible explanations.  
- City-wise analysis: average absolute error per city and suggestions for city-specific modeling.

---

## Deployment plan (Streamlit app)
- Streamlit app (`app/streamlit_app_final.py`) reads `models/ridge_model.pkl` and `scaler.joblib`.  
- App features:
  - Full input form (all features + derived features computed automatically).  
  - Language dropdown (23 languages) — 30-level interpreter translations.  
  - Visualizations: gauge, radar profile, simulation of sleep hour changes, bar for coefficient magnitudes (if coefficients exposed).  
  - Motivational tips & yoga images; download JSON/TXT report.  
- App will include fallback dummy model creation for local testing.

---

## Folder & file checklist (deliverables)
SleepSense_India_Project/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── PROJECT_BLUEPRINT.md
├── data/
│ ├── raw/SleepSense_India_Full.csv
│ └── processed/sleepsense_processed.csv
├── models/ridge_model.pkl
├── notebooks/
│ ├── 00_project_overview.md
│ ├── 01_data_exploration.ipynb
│ ├── 02_data_preprocessing.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_building.ipynb
│ ├── 05_model_evaluation.ipynb
│ ├── 06_model_tuning_and_regularization.ipynb
│ ├── 07_final_interpretation.ipynb
│ └── 08_streamlit_app_preparation.ipynb
├── app/
│ ├── streamlit_app_final.py
│ ├── predict_helper.py
│ └── app_requirements.txt
├── reports/SleepSense_Report.pdf
└── docs/
├── project_abstract.md
└── deployment_guide.md