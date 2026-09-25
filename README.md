# Module 1 — Data Pipeline (`/data_pipeline`)

## Overview
This module automates the extraction, cleaning, currency enrichment, and relational storage of competitive catalog data from `books.toscrape.com`. It builds a production-style, normalized SQLite database and provides both SQL and pandas-based verification workflows.

---

## Currency Conversion Constant
* **Fixed Rate Baseline:** `1 GBP = 105.50 INR`
* As specified in the requirements, this is a fixed, project-defined constant that requires no live API lookup or external date reference. All `price_inr` values are derived from `round(price_gbp * 105.50, 2)`.

---

## Schema Architecture (3NF SQLite)
The relational schema implements a normalized primary key/foreign key structure:
