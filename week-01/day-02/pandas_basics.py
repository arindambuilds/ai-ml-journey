# pandas_basics.py — Day 2 of AI/ML Journey
# Pandas = Panel Data. Used for every real-world dataset in ML.

import pandas as pd
import numpy as np

# ── PART 1: Series — a single column of data ─────────────────

# A Series is like a NumPy array with labels (index)
scores = pd.Series([85, 92, 78, 95, 88],
                   index=["Arindam", "Rahul", "Priya", "Amit", "Sneha"])
print("Scores:\n", scores)
print("\nArindam's score:", scores["Arindam"])
print("Mean score:", scores.mean())
print("Top scorer:", scores.idxmax())

# ── PART 2: DataFrame — a full table of data ─────────────────

# DataFrame = multiple Series together = Excel sheet in Python
data = {
    "Name":       ["Arindam", "Rahul", "Priya", "Amit", "Sneha"],
    "Age":        [28, 24, 26, 30, 25],
    "City":       ["Bhubaneswar", "Kolkata", "Mumbai", "Delhi", "Pune"],
    "ML_Score":   [85, 92, 78, 95, 88],
    "Experience": [3, 1, 2, 5, 2]
}

df = pd.DataFrame(data)
print("\nFull DataFrame:\n", df)

# ── PART 3: Exploring data — first thing you do with any dataset

print("\n--- Basic Info ---")
print("Shape:", df.shape)          # (rows, columns)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("\nFirst 3 rows:\n", df.head(3))
print("\nLast 2 rows:\n", df.tail(2))
print("\nStatistics:\n", df.describe())  # mean, std, min, max etc

# ── PART 4: Selecting data ───────────────────────────────────

print("\n--- Selecting Data ---")
print("Name column:\n", df["Name"])
print("\nName and Score:\n", df[["Name", "ML_Score"]])
print("\nRow 0:\n", df.iloc[0])        # by position
print("\nRow for Rahul:\n", df.loc[1]) # by index

# ── PART 5: Filtering — most used operation in ML ────────────

print("\n--- Filtering ---")
high_scorers = df[df["ML_Score"] > 85]
print("High scorers (>85):\n", high_scorers)

experienced = df[df["Experience"] >= 3]
print("\nExperienced (3+ years):\n", experienced)

# Multiple conditions
senior_high = df[(df["Experience"] >= 2) & (df["ML_Score"] > 85)]
print("\nExperienced AND high score:\n", senior_high)

# ── PART 6: Adding and modifying columns ─────────────────────

df["Score_Per_Year"] = df["ML_Score"] / df["Experience"]
df["Senior"] = df["Experience"] >= 3   # boolean column
print("\nWith new columns:\n", df)

# ── PART 7: Handling missing data — critical in real ML ──────

# Create data with missing values (NaN = Not a Number)
messy_data = {
    "Name":  ["A", "B", "C", "D", "E"],
    "Score": [85, None, 78, None, 88],
    "City":  ["Mumbai", "Delhi", None, "Pune", "Chennai"]
}
messy_df = pd.DataFrame(messy_data)
print("\nMessy data:\n", messy_df)
print("\nMissing values:\n", messy_df.isnull().sum())

# Fix missing values
messy_df["Score"] = messy_df["Score"].fillna(messy_df["Score"].mean())
messy_df["City"] = messy_df["City"].fillna("Unknown")
print("\nCleaned data:\n", messy_df)

# ── PART 8: Groupby — summarise by category ──────────────────

print("\n--- GroupBy ---")
city_avg = df.groupby("City")["ML_Score"].mean()
print("Avg score by city:\n", city_avg)

print("\n--- Day 2 Pandas: Complete ---")

# pandas_basics.py — Day 2 of AI/ML Journey
# Topic: Pandas fundamentals

import pandas as pd
import numpy as np

# 1. Creating a DataFrame manually
data = {
    'name': ['Rahul', 'Priya', 'Arjun', 'Neha', 'Vikram'],
    'age': [22, 25, None, 28, 23],
    'city': ['Bhubaneswar', 'Bangalore', 'Hyderabad', 'Bangalore', None],
    'salary': [35000, 75000, 55000, 90000, 42000],
    'experience_years': [1, 4, 2, 6, 2]
}

df = pd.DataFrame(data)

# 2. First things you always do with any new dataset
print("=== FIRST LOOK ===")
print(df.head())           # first 5 rows
print("\nShape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
# 3. Handling missing data
print("\n=== HANDLING MISSING DATA ===")

# Fill missing age with median (better than mean for small datasets)
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing city with 'Unknown'
df['city'] = df['city'].fillna('Unknown')

print("After filling missing values:")
print(df.isnull().sum())

# 4. Filtering and selecting
print("\n=== FILTERING ===")
bangalore_employees = df[df['city'] == 'Bangalore']
print("Bangalore employees:\n", bangalore_employees)

high_earners = df[df['salary'] > 50000]
print("\nHigh earners:\n", high_earners[['name', 'salary']])

# 5. Adding calculated columns
df['salary_per_year_exp'] = df['salary'] / df['experience_years']
print("\nSalary efficiency:\n", df[['name', 'salary', 'experience_years', 'salary_per_year_exp']])

# 6. Groupby — this is used in almost every ML analysis
print("\n=== GROUPBY ===")
city_avg = df.groupby('city')['salary'].mean()
print("Average salary by city:\n", city_avg)
