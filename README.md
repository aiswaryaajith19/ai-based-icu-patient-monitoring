
# Real-Time Trustworthy AI-Based Patient Monitoring System

**An Explainable AI Approach to Early Detection of Vital-Sign Abnormalities in ICU Patients**

## Overview

This project explores the use of Artificial Intelligence (AI) and Explainable AI (XAI) to identify the risk of future vital-sign abnormalities using early ICU patient data.

The objective is to investigate whether machine learning models can estimate the risk of sustained vital-sign abnormalities and provide explanations that help users understand the factors influencing their predictions.

**Project status:** Work in Progress — Final-Year B.Tech CSE Project

## Objectives

- Prepare and analyse ICU patient vital-sign data.
- Develop baseline machine learning models for predicting future vital-sign abnormalities.
- Evaluate model performance using patient-grouped cross-validation.
- Apply SHAP to interpret model predictions.
- Investigate the influence of missing readings and monitoring patterns.
- Explore a future clinical decision-support dashboard.

## Dataset

Initial experiments use the eICU Collaborative Research Database Demo.

The current experimental setup uses:
- The first six hours of vital-sign data for feature extraction.
- Hours 6–12 to define the future prediction target.
- Heart rate, oxygen saturation (SpO₂), and respiratory rate.

The dataset is not included in this repository.

## Methodology

1. Data preparation and quality checks.
2. Definition and auditing of sustained vital-sign abnormality labels.
3. Feature extraction from early vital-sign measurements.
4. Model training using Logistic Regression and Random Forest.
5. Evaluation using five-fold stratified group cross-validation, grouped by patient.
6. Preliminary model interpretation using SHAP.
7. Investigation of monitoring-frequency and missing-data effects.

## Preliminary Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.648 | 0.633 |
| Average Precision | 0.428 | 0.398 |
| Recall | 0.554 | 0.459 |
| F1-score | 0.450 | 0.419 |

These are preliminary mean cross-validation results. They indicate modest predictive performance and require further investigation.

## Explainable AI

SHAP was used to investigate how features influence model predictions. Early experiments suggest that both physiological summary features and the number of recorded readings contribute to predictions.

The role of missing-data patterns and monitoring frequency remains under investigation.

## Limitations

- The target is a rule-defined sustained vital-sign abnormality, not a clinically confirmed diagnosis.
- The demo dataset is limited in size.
- Predictive performance is modest.
- Future reading availability affects cohort selection.
- The model and explanations have not been clinically validated.

This project is for academic research and experimentation only. It is not intended for clinical use.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib

## Future Work

- Review and refine the prediction target.
- Investigate missing-data patterns and feature reliability.
- Explore additional suitable datasets.
- Improve model evaluation and interpretability.
- Develop a prototype dashboard for presenting predictions and explanations.

## Author

Final-Year B.Tech Computer Science and Engineering Student
