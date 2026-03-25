# real_analysis.py — Day 2: Analysing a real dataset
# Using the Titanic dataset — classic ML beginner dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Titanic dataset directly from URL — no download needed
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== TITANIC DATASET ANALYSIS ===\n")

# ── STEP 1: First look at the data ───────────────────────────
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# ── STEP 2: Missing data check ───────────────────────────────
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(pd.DataFrame({"Missing": missing, "Percent": missing_pct}))

# ── STEP 3: Basic statistics ─────────────────────────────────
print("\n=== STATISTICS ===")
print(df.describe())

# ── STEP 4: Key questions about the data ─────────────────────
print("\n=== SURVIVAL ANALYSIS ===")

# Overall survival rate
survival_rate = df["Survived"].mean() * 100
print(f"Overall survival rate: {survival_rate:.1f}%")

# Survival by gender
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(f"\nSurvival by gender:\n{gender_survival}")

# Survival by passenger class
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(f"\nSurvival by class:\n{class_survival}")

# Average age of survivors vs non-survivors
age_survival = df.groupby("Survived")["Age"].mean()
print(f"\nAverage age (0=died, 1=survived):\n{age_survival}")

# ── STEP 5: Clean the data ───────────────────────────────────
print("\n=== CLEANING ===")

# Fill missing Age with median (better than mean for skewed data)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin — too many missing values (77%)
df = df.drop("Cabin", axis=1)

print("Missing values after cleaning:")
print(df.isnull().sum())

# ── STEP 6: Simple visualisation ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Titanic Dataset Analysis — Day 2", fontsize=14)

# Plot 1: Survival count
df["Survived"].value_counts().plot(kind="bar", ax=axes[0,0],
    color=["#E24B4A", "#1D9E75"])
axes[0,0].set_title("Survival Count")
axes[0,0].set_xticklabels(["Died", "Survived"], rotation=0)

# Plot 2: Survival by gender
gender_survival.plot(kind="bar", ax=axes[0,1], color=["#378ADD", "#D4537E"])
axes[0,1].set_title("Survival Rate by Gender (%)")
axes[0,1].set_xticklabels(["Female", "Male"], rotation=0)

# Plot 3: Age distribution
df["Age"].hist(bins=30, ax=axes[1,0], color="#7F77DD")
axes[1,0].set_title("Age Distribution")
axes[1,0].set_xlabel("Age")

# Plot 4: Survival by class
class_survival.plot(kind="bar", ax=axes[1,1], color="#EF9F27")
axes[1,1].set_title("Survival Rate by Class (%)")
axes[1,1].set_xlabel("Passenger Class")

plt.tight_layout()
plt.savefig("titanic_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as titanic_analysis.png")
print("\n--- Day 2 Real Analysis: Complete ---")
