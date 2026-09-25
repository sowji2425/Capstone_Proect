"""
Zepto Competitive Intelligence Pipeline - Module 1: Data Pipeline
Source: books.toscrape.com
Flow: Scrape -> Clean & Impute -> Enrich (GBP to INR) -> Relational SQLite -> SQL & Pandas Verification
"""

import re
import sqlite3
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50  # Required fixed project-defined baseline conversion constant
DB_PATH = "books_pipeline.db"

# Scrape across 4 distinct categories (guarantees >= 60 items across >= 3 categories)
TARGET_CATEGORIES = [
    {"name": "Travel", "url": "catalogue/category/books/travel_2/index.html"},
    {"name": "Mystery", "url": "catalogue/category/books/mystery_3/index.html"},
    {"name": "Historical Fiction", "url": "catalogue/category/books/historical-fiction_4/index.html"},
    {"name": "Sequential Art", "url": "catalogue/category/books/sequential-art_5/index.html"}
]

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

# ==========================================
# STEP 1: SCRAPING
# ==========================================
def scrape_category_books(category_name: str, relative_url: str, session: requests.Session) -> List[Dict[str, Any]]:
    """
    Scrapes all book items for a given category, handling category pagination if present.
    """
    category_books = []
    current_url = BASE_URL + relative_url

    while current_url:
        print(f"[{category_name}] Fetching: {current_url}")
        resp = session.get(current_url, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        articles = soup.select("article.product_pod")

        for art in articles:
            # Title: fallback to text if title attribute is omitted
            title_tag = art.select_one("h3 a")
            raw_title = title_tag.get("title") or title_tag.text.strip() if title_tag else None

            # Raw Price text
            price_tag = art.select_one("p.price_color")
            raw_price = price_tag.text.strip() if price_tag else None

            # Star rating text from class (e.g. "star-rating Three")
            rating_tag = art.select_one("p.star-rating")
            raw_rating = None
            if rating_tag:
                classes = rating_tag.get("class", [])
                rating_classes = [c for c in classes if c.lower() != "star-rating"]
                raw_rating = rating_classes[0] if rating_classes else None

            # Availability text
            avail_tag = art.select_one("p.instock.availability")
            raw_availability = avail_tag.text.strip() if avail_tag else None

            category_books.append({
                "title": raw_title,
                "raw_price": raw_price,
                "raw_rating": raw_rating,
                "raw_availability": raw_availability,
                "category": category_name
            })

        # Check for next page inside the category
        next_button = soup.select_one("li.next a")
        if next_button:
            next_page_rel = next_button.get("href")
            # Build relative link based on category base folder
            base_cat_path = current_url.rsplit("/", 1)[0]
            current_url = f"{base_cat_path}/{next_page_rel}"
        else:
            current_url = None

    return category_books


def scrape_catalog() -> pd.DataFrame:
    """Scrapes all designated categories and returns a raw DataFrame."""
    session = requests.Session()
    session.headers.update({"User-Agent": "ZeptoCatalogBenchmark/1.0 (DataPipeline Exercise)"})

    all_records = []
    for cat in TARGET_CATEGORIES:
        records = scrape_category_books(cat["name"], cat["url"], session)
        all_records.extend(records)

    raw_df = pd.DataFrame(all_records)
    print(f"\n[Scraping Complete] Collected {len(raw_df)} records across {raw_df['category'].nunique()} categories.")
    return raw_df

# ==========================================
# STEP 2: CLEANING & ENRICHMENT
# ==========================================
def clean_and_enrich_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw columns:
      - price_gbp: regex match float; impute missing/malformed with median price
      - rating: mapped One..Five -> 1..5; impute missing with median rating
      - in_stock: parse text for 'in stock' -> boolean int (0 or 1)
      - price_inr: enriched using 1 GBP = 105.50 INR baseline
    """
    df = raw_df.copy()

    # Clean Price -> Float
    def parse_price(val: Optional[str]) -> Optional[float]:
        if not val:
            return None
        match = re.search(r"(\d+\.\d+|\d+)", val)
        return float(match.group(1)) if match else None

    df["price_gbp"] = df["raw_price"].apply(parse_price)

    # Impute missing price using category median (or global median if category is empty)
    price_median = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(price_median).round(2)

    # Clean Rating -> Integer (1-5)
    def parse_rating(val: Optional[str]) -> Optional[int]:
        if not val:
            return None
        return RATING_MAP.get(val.strip().lower(), None)

    df["rating"] = df["raw_rating"].apply(parse_rating)
    rating_median = int(df["rating"].median()) if not df["rating"].dropna().empty else 3
    df["rating"] = df["rating"].fillna(rating_median).astype(int)

    # Clean Availability -> Boolean (stored as int 0/1 for SQLite compatibility)
    df["in_stock"] = df["raw_availability"].str.lower().str.contains("in stock").fillna(False).astype(int)

    # Filter out rows missing essential identity (drop row if title is missing)
    initial_len = len(df)
    df = df.dropna(subset=["title"]).copy()
    if len(df) < initial_len:
        print(f"[Cleaning] Dropped {initial_len - len(df)} row(s) missing product title.")

    # Enrichment: Calculate price_inr via fixed rate baseline
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    # Retain final structured columns
    cleaned_df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].reset_index(drop=True)
    return cleaned_df

# ==========================================
# STEP 3: DATABASE SCHEMA & LOAD
# ==========================================
def initialize_database(db_path: str = DB_PATH):
    """Initializes SQLite database with normalized 3NF schema and foreign key enforcement."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        cursor.executescript("""
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE RESTRICT
            );
        """)
        conn.commit()


def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    """Loads normalized data into categories and books tables."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        # 1. Insert Categories
        unique_categories = sorted(df["category"].unique())
        cursor.executemany(
            "INSERT INTO categories (category_name) VALUES (?);",
            [(cat,) for cat in unique_categories]
        )

        # Retrieve mapped category_id values
        category_map = {
            row[0]: row[1]
            for row in cursor.execute("SELECT category_name, category_id FROM categories;").fetchall()
        }

        # 2. Insert Books with foreign key mapping
        books_payload = [
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                row["in_stock"],
                category_map[row["category"]]
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            books_payload
        )
        conn.commit()

    print(f"[Database Load] Inserted {len(unique_categories)} categories and {len(books_payload)} books into '{db_path}'.")

# ==========================================
# STEP 4: SQL QUERIES EXECUTION
# ==========================================
SQL_QUERIES = [
    {
        "id": "Q1",
        "description": "SELECT with WHERE and ORDER BY (High-value books priced over £40 sorted descending)",
        "query": """
            SELECT title, price_gbp, price_inr, rating
            FROM books
            WHERE price_gbp > 40.0
            ORDER BY price_gbp DESC;
        """
    },
    {
        "id": "Q2",
        "description": "LIMIT with ORDER BY (Top 5 most affordable in-stock titles)",
        "query": """
            SELECT title, price_gbp, price_inr, in_stock
            FROM books
            WHERE in_stock = 1
            ORDER BY price_gbp ASC
            LIMIT 5;
        """
    },
    {
        "id": "Q3",
        "description": "DISTINCT rating values currently populated across all inventory",
        "query": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating DESC;
        """
    },
    {
        "id": "Q4",
        "description": "BETWEEN and IN clauses (Books rated 4 or 5 stars with INR price between 2000 and 4000)",
        "query": """
            SELECT title, rating, price_inr
            FROM books
            WHERE rating IN (4, 5)
              AND price_inr BETWEEN 2000.0 AND 4000.0
            ORDER BY price_inr DESC;
        """
    },
    {
        "id": "Q5 (Primary Join)",
        "description": "INNER JOIN across categories and books: Top 10 highest-rated in-stock books with category names",
        "query": """
            SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating
            FROM books b
            INNER JOIN categories c ON b.category_id = c.category_id
            WHERE b.in_stock = 1
            ORDER BY b.rating DESC, b.price_gbp DESC
            LIMIT 10;
        """
    }
]

def run_sql_queries(db_path: str = DB_PATH):
    """Executes each SQL query, printing formatted results."""
    print("\n" + "="*80)
    print("EXECUTING MANDATORY SQL QUERIES")
    print("="*80)

    with sqlite3.connect(db_path) as conn:
        for item in SQL_QUERIES:
            print(f"\n--- {item['id']}: {item['description']} ---")
            print("SQL Statement:")
            print(item['query'].strip())
            df_result = pd.read_sql_query(item['query'], conn)
            print(f"\nResult ({len(df_result)} rows):")
            print(df_result.to_string(index=False))

# ==========================================
# STEP 5: PANDAS EQUIVALENCE VERIFICATION
# ==========================================
def verify_pandas_vs_sql(db_path: str = DB_PATH):
    """
    Reads query results via pd.read_sql and independently reproduces
    the relational join using pure pd.merge in memory. Demonstrates exact equivalence.
    """
    print("\n" + "="*80)
    print("VERIFYING pd.read_sql vs pd.merge EQUIVALENCE")
    print("="*80)

    with sqlite3.connect(db_path) as conn:
        # 1. SQL Result for Q5
        sql_join_query = SQL_QUERIES[4]["query"]
        sql_result_df = pd.read_sql_query(sql_join_query, conn)

        # 2. In-Memory Pandas Merge
        books_df = pd.read_sql_query("SELECT * FROM books;", conn)
        categories_df = pd.read_sql_query("SELECT * FROM categories;", conn)

    # Perform pandas merge recreating the exact SQL logic:
    # INNER JOIN ON category_id, filter in_stock == 1, sort rating desc / price_gbp desc, head(10)
    merged_df = (
        pd.merge(books_df, categories_df, on="category_id", how="inner")
        .query("in_stock == 1")
        .sort_values(by=["rating", "price_gbp"], ascending=[False, False])
        [["title", "category_name", "price_gbp", "price_inr", "rating"]]
        .head(10)
        .reset_index(drop=True)
    )

    print("\n[Output A] pd.read_sql result:")
    print(sql_result_df)

    print("\n[Output B] Pure pd.merge equivalent result:")
    print(merged_df)

    # Assert exact frame match
    try:
        pd.testing.assert_frame_equal(sql_result_df, merged_df, check_dtype=True)
        print("\n✔ SUCCESS: pd.read_sql and pd.merge produce identical DataFrames!")
    except AssertionError as err:
        print("\n❌ Equivalence check failed:", err)


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("Starting Zepto Competitive Intelligence Data Pipeline...")
    
    # 1. Scrape
    raw_df = scrape_catalog()
    assert len(raw_df) >= 60, f"Expected >= 60 records, got {len(raw_df)}"
    assert raw_df["category"].nunique() >= 3, f"Expected >= 3 categories, got {raw_df['category'].nunique()}"

    # 2. Clean & Enrich
    clean_df = clean_and_enrich_data(raw_df)

    # 3. Setup SQLite and Load
    initialize_database(DB_PATH)
    load_to_sqlite(clean_df, DB_PATH)

    # 4. Run SQL Queries
    run_sql_queries(DB_PATH)

    # 5. Verify SQL vs Pandas
    verify_pandas_vs_sql(DB_PATH)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
