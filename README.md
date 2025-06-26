# Heart Disease Classification Project

This project focuses on predicting the presence of heart disease using a variety of machine learning models. The dataset includes clinical and personal health features, such as cholesterol, resting blood pressure, and more. The notebook applies preprocessing, feature handling, model training, and evaluation to compare several classifiers.

---

## Objectives

- Identify the most accurate classification model for heart disease prediction.
- Clean and preprocess real-world medical data effectively.
- Apply proper outlier handling, skewness correction, scaling, and model evaluation techniques.
- Explore the performance impact of hyperparameter tuning and ensemble learning.

---

## Dataset Overview

The dataset includes 918 samples with the following key features:

- **Age**: Age of the patient (years)
- **Sex**: Sex of the patient (M: Male, F: Female)
- **ChestPainType**: Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- **RestingBP**: Resting blood pressure (mmHg)
- **Cholesterol**: Serum cholesterol (mg/dL)
- **FastingBS**: Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
- **RestingECG**: Resting electrocardiogram results (Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y: Yes, N: No)
- **Oldpeak**: ST depression induced by exercise
- **ST_Slope**: the slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
- **HeartDisease**: Binary target (0 = No, 1 = Yes)

---

## Preprocessing Steps

1. **Missing/Zero Handling**:
   No missing values detected.

2. **Outlier Detection & Handling**:
   - Used IQR-based method to detect outliers.
   - Applied conditional winsorization depending on outlier severity:
     - Mild: Winsorize at 1%
     - Moderate: Winsorize at 5%

3. **Skewness Handling**:
   - Used `boxcox1p()` or `log1p()` to reduce skew in highly skewed features.
   - Applied only to continuous features with significant skewness.

4. **Scaling**:
   - Applied `StandardScaler` after all transformations.

5. **Encoding**:
   - One-hot encoded categorical features where applicable.

---

## Models Used

### Baseline Models (No Tuning):
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting Classifier

### Tuned Models with customized hyperparameters:
- Tuned SVM 
- Tuned KNN
- Tuned Gradient Boosting

### Ensemble:
- Stacking Classifier combining the best-performing tuned models

---

## Evaluation Metrics

- **Accuracy**: Main focus of optimization
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- All metrics computed on a **fixed train/test split** (with `stratify`) or using **KFold cross-validation** depending on the experiment

---

## Visualization

- Skewness histograms before and after transformation
- Correlation heatmap of features
- Confusion matrices for all models
- Metric comparison bar plots

---

## Tools & Libraries

- Python 3.12
- `pandas`, `numpy`, `matplotlib`, `seaborn` (for loading data, preprocessing, visualization)
- `scikit-learn` (for feature engineering and training models)
- `scipy` (for `boxcox` and winsorization)

---

## Future Work

- Try other ensemble techniques like Random Forest or XGBoost
- Incorporate external health data (e.g. lifestyle, ECG)
- Build a small Streamlit or Flask app for prediction
