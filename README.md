# SleepSense-Predictor
🌙 AI-powered Sleep Quality Predictor using real-world lifestyle & environmental factors (screen time, noise, AQI, stress). Built with Python, ML, and Streamlit. Supports 22 Indian languages.

# 💤 SleepSense — India Edition  
### Predicting Sleep Quality of Indians using Machine Learning

---

## 📘 Overview
**SleepSense India** is a machine learning project that predicts the **Sleep Quality Score (0–100)** based on lifestyle and environmental factors.  
The work focuses on understanding how personal habits such as screen time, caffeine intake, and stress influence sleep quality in the Indian context.

The project follows a clear supervised learning pipeline using **Linear Regression** and demonstrates every concept step-by-step from data preparation to model evaluation and deployment.

---

## 🎯 Objectives
- Apply **Supervised Learning (Regression)** for prediction  
- Build and train a **Linear Regression** model using real-world data  
- Compare model performance using **R²** and **Adjusted R²**  
- Perform **Feature Engineering** and **Regularization (Ridge & Lasso)**  
- Create an interactive **Streamlit app** for real-time prediction  

---

## 📊 Dataset
**File:** `data/raw/SleepSense_India_Full.csv`  
**Rows:** 12,000  **Columns:** 22  

**Feature Categories**
- **Personal:** `age`, `sex`, `family_size`  
- **Lifestyle:** `tea_cups`, `coffee_cups`, `screen_time_hours`, `stress_level`, `physical_activity_min`  
- **Environmental:** `city`, `city_noise_dB`, `light_pollution_index`, `air_quality_index`, `temperature_night`, `humidity_night`  
- **Target:** `sleep_quality_score` (0–100)

The dataset represents a variety of Indian cities and cultural patterns related to sleep behavior.

---

## 🧠 Project Workflow

| Step | Notebook | Description |
|------|-----------|--------------|
| 00 | `00_project_overview.ipynb` | Introduction and theoretical background |
| 01 | `01_data_exploration.ipynb` | Exploratory Data Analysis (EDA) |
| 02 | `02_data_preprocessing.ipynb` | Data cleaning, scaling, and encoding |
| 03 | `03_feature_engineering.ipynb` | Creation of new meaningful features |
| 04 | `04_model_building.ipynb` | Model training using Linear Regression |
| 05 | `05_model_evaluation.ipynb` | Evaluation using R², Adjusted R², RMSE |
| 06 | `06_model_tuning_regularization.ipynb` | Ridge and Lasso regularization |
| 07 | `07_final_interpretation_visuals.ipynb` | Insights and visualization of results |
| 08 | `08_streamlit_app_preparation.ipynb` | Model export and app setup |
| 09 | `09_summary_and_export.ipynb` | Summary report and visual export |

---

## 🧰 Tools and Libraries
- **Python 3.10+**
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `joblib`

All steps follow standard supervised learning concepts covered in academic curriculum.

---

## ⚙️ Setup and Execution

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
