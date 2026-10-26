# Heart-Disease-Prediction-ML
Heart Disease Prediction using Machine Learning

A machine learning project that predicts whether a patient has heart disease from clinical measurements, and compares five classification models in Python.

# Problem

Heart disease is found by combining several test results. Can a model learn from past patient records and predict heart disease for a new patient? I trained five different classifiers and compared them.

# Dataset
File: heart.csv, 918 patients and 12 columns (11 features and the target HeartDisease)
No missing values and no duplicate rows. About 55% of the patients have heart disease (class 1).
Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

# Approach
Exploratory data analysis — shape, data types, summary statistics, missing and duplicate checks; histograms, count plots by heart disease, box plots, a violin plot and a correlation heatmap
Data cleaning — Cholesterol had 172 zero values and RestingBP had a zero, which are not possible medical readings. I treated them as missing and replaced them with the mean of the non-zero values.
Encoding — one-hot encoded the categorical columns (get_dummies, drop_first=True)
Scaling — standardised the numeric columns with StandardScaler
Split — 80% training and 20% testing (random_state=42)
Models — Logistic Regression, KNN, Naive Bayes, Decision Tree and SVM, all with default settings
Evaluation — accuracy and F1 score on the test set
Saving — the KNN model, the scaler and the column list are saved with joblib
# Results
Model	Accuracy	F1 score
Logistic Regression	0.8696	0.8857
KNN	0.8641	0.8815
Naive Bayes	0.8533	0.8683
SVM	0.8478	0.8679
Decision Tree	0.7772	0.7980

Logistic Regression performed best on this test set, and the single Decision Tree performed worst. The simpler models did as well as or better than the others on this small dataset.

# Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter Notebook

# Limitations and next steps
The dataset is small (918 patients), so the test set has only 184 patients and the scores can change with a different split.
Models use default settings. Next: cross-validation and hyperparameter tuning.
For a medical problem, missing a sick patient is costly. Next: report recall, precision and a confusion matrix, not only accuracy and F1.
Try ensemble models such as Random Forest and Gradient Boosting.
Fit the scaler on the training data only and apply it to the test data, so no test information leaks into training.
This is a learning project and not a medical tool.
# How to run
bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook ML_project2_.ipynb
# Author
Arpita Singh
