# Zepto Data & AI Guild — Connected AI/ML Platform
### Capstone Project Submission

This repository contains a unified, three-module platform developed as an incoming AI/ML engineer for Zepto's analytics guild. The platform runs entirely within a single repository, leveraging an integrated environment configuration to connect data engineering, predictive modeling, and grounded GenAI services.

---

## 1. Project Setup & Installation

### Environment Configuration Strategy
For this project, a **single consolidated `requirements.txt`** located at the repository root was chosen rather than separate module-level configuration files. This strategy prevents package dependency conflicts between modules (such as version mismatches between `pandas` usage in data pipelines and model pipelines) and provides a clean, single-step initialization for developers and evaluators.

### Deployment Instructions
1. Clone this repository to your local machine.
2. Navigate to the root directory of the repository.
3. Install the unified environment profile by executing:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. End-to-End Execution Guide

### Module 1: Data Engineering Pipeline (`/data_pipeline`)
This module crawls the public target catalog website, normalizes the raw attributes, performs a constant currency conversion, and stores structured elements into an SQLite instance.
* **Step 1:** Run the web scraper to extract product listings and parse raw attributes:
  ```bash
  python data_pipeline/scraper.py
  ```
* **Step 2:** Initialize the database schema constraints and load the structured dataset:
  ```bash
  python data_pipeline/database.py
  ```
* **Step 3:** Run the analytical extraction script to perform required SQL operations (`SELECT/WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN/IN`, and `JOIN`) and load tables into pandas:
  ```bash
  python data_pipeline/queries.py
  ```

### Module 2: Analytics & Predictive Pipeline (`/analytics`)
This module processes real-world behavioral outcomes using the benchmark Titanic passenger matrix to execute data profiling, clean sparse values, and train classification models.
* **Step 1:** Generate the profile analysis and verify baseline missingness:
  ```bash
  python analytics/data_profiler.py
  ```
* **Step 2:** Process categorical features, scale parameters, and evaluate model performance:
  ```bash
  python analytics/model_pipeline.py
  ```

### Module 3: GenAI Grounded Support Assistant (`/support_assistant`)
This module launches a terminal interface that allows users to query internal Zepto policies. Queries are processed through a grounded Retrieval-Augmented Generation (RAG) architecture.
* **Step 1:** Start the interactive support tool session:
  ```bash
  python support_assistant/app.py
  ```
* **Step 2:** Type policy questions (e.g., questions containing keywords like *"refund"* or *"10 minutes"*) into the interactive prompt. Type `exit` or `quit` to close the session.

---

## 3. Core Design Decisions & Justifications

### Module 1 — Data Pipeline
* **Missing Field Management Strategy:** Rows with invalid or missing text ratings or prices are dropped rather than filled with estimated values. In competitive e-commerce catalog tracking, using fabricated pricing trends introduces downstream accounting risks. Dropping malformed items protects database integrity.
* **Database Normalization Schema:** A strict two-table design (`categories` and `books`) linked via an enforced Foreign Key constraint was implemented. This normalizes repeated text elements into individual keys, reducing the overall storage footprint.

### Module 2 — Analytics & Modeling
* **Data Selection & Cleaning Strategy:** The workflow uses a real-world dataset instead of synthetic mock text blocks to ensure valid evaluation metrics. Missing continuous values (like `Age`) are filled with grouped median parameters to protect the natural data distribution, while high-sparsity entries (like `Cabin`) are removed to avoid overfitting.
* **Model Selection:** A Random Forest Classifier architecture was selected. This model handles mixed structural parameters (such as a combination of categorical textual items and scaled continuous numeric ranges) without requiring strict linear relationships.

### Module 3 — GenAI Support Assistant
* **Grounded Retrieval Strategy:** To fulfill the project's requirement to operate without paid external services, the engine uses a local token-overlap vector lookup pattern. This ensures text matching remains highly accurate and reproducible across test systems without requiring live internet connections or commercial API subscriptions.
* **Hallucination Prevention:** The prompt template includes strict rules that force the engine to reply with a transparent fallback statement if a user's question cannot be matched to the internal context document. This blocks ungrounded fabrications completely.
