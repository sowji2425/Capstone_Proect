# Module 2 — Analytics Pipeline (`/analytics`)

## Overview
This module implements a complete, end-to-end data science workflow on the Titanic dataset, spanning defensive data profiling, IQR outlier detection, bivariate/multivariate visual analysis, pipeline-enforced ML modeling, imbalance handling, hyperparameter tuning with OOB evaluation, a regression side-task, and artifact serialization.

---

## Part A: EDA & Data Cleaning Summary
* **Offline Fallback Dataset:** Successfully committed as `titanic.csv` via `df.to_csv("titanic.csv", index=False)` upon initial load.
* **Missing Value Threshold Strategy:**
  * `deck` (~77.22% missing): **Dropped column** (exceeded 30% threshold; imputation would introduce severe synthetic noise).
  * `embarked` / `embark_town` (~0.22% missing): **Dropped rows** (< 5% threshold rule).
  * `age` (~19.87% missing): **Imputed with median** (falls cleanly within the 5%–30% imputation threshold).
* **Univariate & Outliers (IQR Rule):**
  * **Age:** Q1 = 22.0, Q3 = 35.0, IQR = 13.0. Outlier bounds: [-1.5, 58.5]. Outlier count: **11 rows**.
  * **Fare:** Q1 = 7.91, Q3 = 31.0, IQR = 23.09. Outlier bounds: [-26.72, 65.63]. Outlier count: **116 rows**.
  * **Fare Skewness:** Mean (`32.20`) > Median (`14.45`) > Mode (`8.05`). **Conclusion:** Strongly **right-skewed**.
* **Bivariate Survival Breakdowns:**
  * **By Sex:** Female survival rate = **74.2%**, Male survival rate = **18.9%**.
  * **By Pclass:** 1st Class = **62.6%**, 2nd Class = **47.3%**, 3rd Class = **24.2%**.
  * **By Sex & Pclass:** Female 1st = **96.8%**, Female 3rd = **50.0%**, Male 1st = **36.9%**, Male 3rd = **13.5%**.
* **6x6 Correlation Matrix (Top 2 Off-Diagonal Pairs):**
  1. `pclass` & `fare` (Absolute correlation coefficient: **`0.55`**): Higher passenger classes cost substantially more.
  2. `age` & `pclass` (Absolute correlation coefficient: **`0.37`**): Older passengers tended to travel in higher classes.

---

## Part B: Model Comparison Table
Classification and regression metrics operate on different scales and are presented below as two distinct groups:

| Model Name | Accuracy | Precision | Recall | F1 Score | AUC | Regression Model | MAE | RMSE | R² | Adjusted R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.810 | 0.776 | 0.730 | 0.752 | 0.851 | **Multivariate Linear Regression (Fare)** | 10.42 | 14.85 | 0.412 | 0.398 |
| **Decision Tree (max_depth=4)**| 0.821 | 0.810 | 0.703 | 0.753 | 0.832 | — | — | — | — | — |
| **Random Forest (Tuned)** | **0.838** | **0.812** | **0.770** | **0.791** | **0.865** | — | — | — | — | — |

---

## Final Recommendation & Deployment Note
**Recommendation:** Deploy the tuned **Random Forest Classifier** (`best_titanic_pipeline.joblib`). It achieves the highest overall accuracy (**0.838**), best AUC (**0.865**), and a well-balanced F1 score (**0.791** with high recall of 0.770). Unlike Logistic Regression which assumes linear feature boundaries, Random Forest effectively captures non-linear interactions between passenger class, sex, and fare.

**Artifact Verification:** The saved joblib bundle encapsulates both the `ColumnTransformer` preprocessing steps and the tuned classifier estimator together, allowing seamless end-to-end predictions directly on raw, unpreprocessed inputs.
