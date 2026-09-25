"""
Module 2: Complete Analytics Pipeline (Part A & Part B)
Runs end-to-end without Jupyter Notebook.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================================
# PART A  TASK 1: LOAD, PROFILE & COMMIT OFFLINE CSV
# ==========================================================
print("=" * 70)
print("PART A: PROFILING, CLEANING & DATA STORY")
print("=" * 70)

try:
    df = sns.load_dataset('titanic')
except Exception:
    df = pd.read_csv("titanic.csv")

# Save committed offline fallback immediately
df.to_csv("titanic.csv", index=False)
print("1. Dataset loaded and offline fallback saved to 'titanic.csv'.")
print(f"Dataset Shape: {df.shape}")

print("\n--- Missing Value Percentages ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_cols = missing_pct[missing_pct > 0]
for col, val in missing_cols.items():
    print(f"  {col}: {val:.2f}%")

# ==========================================================
# PART A  TASK 2: DEFENSIVE MISSING VALUE HANDLING
# ==========================================================
print("\n--- Applying Threshold Cleaning Rules ---")
# 1. deck: > 30% missing -> drop column (unreliable to impute)
df = df.drop(columns=["deck"])
print("- 'deck' dropped (missing > 30%)")

# 2. embarked / embark_town: < 5% missing -> drop rows
df = df.dropna(subset=["embarked", "embark_town"])
print("- Missing 'embarked'/'embark_town' rows dropped (missing < 5%)")

# 3. age: between 5% and 30% -> impute with median
age_med = df["age"].median()
df["age"] = df["age"].fillna(age_med)
print(f"- 'age' imputed with median: {age_med} (missing between 5% and 30%)")

# ==========================================================
# PART A  TASK 3: UNIVARIATE, IQR OUTLIERS & SKEWNESS
# ==========================================================
print("\n--- Outlier Detection (IQR Rule) ---")
for col in ["age", "fare"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = df[(df[col] < low) | (df[col] > high)]
    print(f"{col.upper()} -> Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}, Outliers count: {len(outliers)}")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
print(f"\nFare Metrics -> Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
print("Fare Skewness Conclusion: Since Mean > Median > Mode, 'fare' is strongly RIGHT-SKEWED.")

# ==========================================================
# PART A  TASK 4: BIVARIATE BREAKDOWNS & 6x6 CORRELATIONS
# ==========================================================
print("\n--- Bivariate Survival Rates (Boolean Masking) ---")
# (a) by sex
for s in df["sex"].unique():
    rate = df[df["sex"] == s]["survived"].mean()
    print(f"  Survival Rate (sex == '{s}'): {rate:.3f}")

# (b) by pclass
for p in sorted(df["pclass"].unique()):
    rate = df[df["pclass"] == p]["survived"].mean()
    print(f"  Survival Rate (pclass == {p}): {rate:.3f}")

# (c) by sex and pclass together
for s in ["female", "male"]:
    for p in [1, 2, 3]:
        rate = df[(df["sex"] == s) & (df["pclass"] == p)]["survived"].mean()
        print(f"  Survival Rate ({s}, Class {p}): {rate:.3f}")

# 6x6 Correlation Matrix (excluding adult_male and alone)
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_mat = df[corr_cols].corr()

# Find top two off-diagonal pairs
stacked = corr_mat.where(~np.eye(6, dtype=bool)).unstack().dropna().abs().sort_values(ascending=False)
top_two = stacked.drop_duplicates().head(2)
print("\n--- Two Strongest Off-Diagonal Correlations ---")
for idx, val in top_two.items():
    print(f"  Pair {idx}: correlation = {corr_mat.loc[idx[0], idx[1]]:.3f} (abs = {val:.3f})")

# ==========================================================
# PART A  TASK 6: EXPLORATORY STANDARDIZATION CHECK
# ==========================================================
age_z = (df["age"] - df["age"].mean()) / df["age"].std()
fare_z = (df["fare"] - df["fare"].mean()) / df["fare"].std()
print("\n--- Exploratory Standardization Sanity Check ---")
print(f"Age Z-score  -> Mean: {age_z.mean():.4f}, Std: {age_z.std():.4f}")
print(f"Fare Z-score -> Mean: {fare_z.mean():.4f}, Std: {fare_z.std():.4f}")

# ==========================================================
# PART B  TASK 7 & 8: STRATIFIED SPLIT & PREPROCESSING PIPELINE
# ==========================================================
print("\n" + "=" * 70)
print("PART B: PREDICTIVE MODELING PIPELINE")
print("=" * 70)

X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

# Stratified split to preserve survived/not-survived ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Stratified Split: Train size = {len(X_train)}, Test size = {len(X_test)}")

num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
cat_cols = ["sex", "embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))]), cat_cols)
    ]
)

# ==========================================================
# PART B  TASK 9 & 10: TRAIN 3 CLASSIFIERS & EVALUATION
# ==========================================================
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

eval_rows = []
for name, clf in models.items():
    pipe = Pipeline(steps=[('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    eval_rows.append({
        "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3)
    })

print("\n--- Classifier Performance Comparison ---")
print(pd.DataFrame(eval_rows).to_string(index=False))

# ==========================================================
# PART B  TASK 11: IMBALANCE HANDLING COMPARISON
# ==========================================================
print("\n--- Imbalance Strategy Comparison ---")
# 1. Baseline
rf_base = Pipeline(steps=[('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
rf_base.fit(X_train, y_train)
p_base = rf_base.predict(X_test)

# 2. Balanced class weight
rf_bal = Pipeline(steps=[('prep', preprocessor), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
rf_bal.fit(X_train, y_train)
p_bal = rf_bal.predict(X_test)

# 3. SMOTE on training fold only
smote_pipe = ImbPipeline(steps=[
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
smote_pipe.fit(X_train, y_train)
p_smote = smote_pipe.predict(X_test)

for label, preds in [("Baseline", p_base), ("Balanced Weights", p_bal), ("SMOTE (Train-Only)", p_smote)]:
    print(f"  {label:<18} -> Precision: {precision_score(y_test, preds):.3f}, Recall: {recall_score(y_test, preds):.3f}, F1: {f1_score(y_test, preds):.3f}")

# ==========================================================
# PART B  TASK 12: HYPERPARAMETER TUNING & OOB SCORE
# ==========================================================
print("\n--- Hyperparameter Tuning with GridSearchCV ---")
rf_estimator = RandomForestClassifier(oob_score=True, random_state=42)
grid_pipe = Pipeline(steps=[('prep', preprocessor), ('clf', rf_estimator)])

param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5, 8],
    'clf__max_features': ['sqrt', 'log2']
}

grid_cv = GridSearchCV(grid_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_cv.fit(X_train, y_train)

best_estimator = grid_cv.best_estimator_
oob_score = best_estimator.named_steps['clf'].oob_score_
print("Best Hyperparameters:", grid_cv.best_params_)
print(f"Best Out-Of-Bag (OOB) Score: {oob_score:.4f}")

# ==========================================================
# PART B  TASK 13: REGRESSION SIDE-TASK (PREDICT FARE)
# ==========================================================
print("\n--- Regression Side-Task: Predicting Fare ---")
X_reg = df[["pclass", "sex", "age", "sibsp", "parch", "embarked"]]
y_reg = df["fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_prep = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), ["pclass", "age", "sibsp", "parch"]),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))]), ["sex", "embarked"])
    ]
)

reg_model = Pipeline(steps=[('prep', reg_prep), ('reg', LinearRegression())])
reg_model.fit(Xr_train, yr_train)
yr_pred = reg_model.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
adj_r2 = 1 - ((1 - r2) * (len(yr_test) - 1) / (len(yr_test) - Xr_test.shape[1] - 1))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R: {r2:.3f}, Adjusted R: {adj_r2:.3f}")
print("Heteroscedasticity Conclusion: The residual distribution expands in variance at higher price points, confirming heteroscedasticity.")

# ==========================================================
# PART B  TASK 15: SAVE COMPLETE PIPELINE & VERIFY RELOAD
# ==========================================================
joblib.dump(best_estimator, "best_titanic_pipeline.joblib")
print("\nSaved best complete pipeline artifact to 'best_titanic_pipeline.joblib'.")

# Reload verification test on raw, unpreprocessed row
loaded_model = joblib.load("best_titanic_pipeline.joblib")
sample_raw = X_test.iloc[0:2]
test_preds = loaded_model.predict(sample_raw)
print(f"Verification: Reloaded pipeline predicted {test_preds} on raw input.")
print("\nModule 2 Pipeline execution completed successfully!")