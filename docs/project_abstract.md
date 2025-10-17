# SleepSense — Project Abstract

## Title
**SleepSense: Predicting Sleep Quality Using Environmental and Lifestyle Factors (India Edition)**

## Abstract
SleepSense is a machine learning project designed to estimate a person’s **sleep quality score (0–100)** based on multiple lifestyle, environmental, and digital behavior factors.  
The motivation comes from growing sleep issues among Indian students and professionals caused by high screen time, urban noise, irregular routines, and stress.

The dataset combines both **realistic and synthetic** data representing Indian urban and rural populations. Using features like **average sleep hours, city noise, light pollution, caffeine intake, and physical activity**, a supervised regression model predicts a numeric sleep quality score.

Three models were tested — **Linear Regression, Ridge Regression, and Lasso Regression**. Ridge Regression performed best with an R² of 0.46, RMSE of 6.36.  
The project also includes an interactive **Streamlit web app** that allows users to enter their daily habits, choose their language (22 Indian languages supported), and instantly view their predicted score along with lifestyle suggestions, yoga poses, and motivational feedback.

## Key Outcomes
- Built an end-to-end supervised learning pipeline: data engineering → feature design → modeling → evaluation → deployment.  
- Designed multilingual interface for accessibility across Indian regions.  
- Integrated **realistic city-level environment variables (AQI, noise, light pollution)**.  
- Demonstrated model interpretability via coefficient analysis and residual diagnostics.

## Keywords
Sleep Quality, Regression, Lifestyle Analytics, Streamlit, Multilingual AI, Ridge Regression, India-specific Dataset
