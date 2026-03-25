# seaborn_basics.py — Day 3 of AI/ML Journey
# Topic: Seaborn for statistical visualisation

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Seaborn comes with built-in datasets — perfect for learning
# This is the famous Titanic dataset
titanic = sns.load_dataset('titanic')

print("Dataset loaded.")
print(f"Shape: {titanic.shape}")
print(f"\nFirst look:\n{titanic.head()}")
print(f"\nColumns: {list(titanic.columns)}")
print(f"\nMissing values:\n{titanic.isnull().sum()}")
# ── CHART 1: Survival by Sex ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=titanic, x='sex', hue='survived', 
              palette=['#E74C3C', '#2ECC71'], ax=ax)
ax.set_title('Titanic Survival Count by Sex', fontsize=14, fontweight='bold')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.legend(['Did not survive', 'Survived'])
plt.tight_layout()
plt.savefig('outputs/05_survival_by_sex.png', dpi=150)
plt.close()
print("Saved: outputs/05_survival_by_sex.png")

# ── CHART 2: Age Distribution by Survival ─────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=titanic, x='age', hue='survived', 
             bins=30, kde=True, palette=['#E74C3C', '#2ECC71'], ax=ax)
ax.set_title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/06_age_by_survival.png', dpi=150)
plt.close()
print("Saved: outputs/06_age_by_survival.png")

# ── CHART 3: Fare Distribution by Class ───────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=titanic, x='pclass', y='fare', 
            palette='viridis', ax=ax)
ax.set_title('Fare Distribution by Passenger Class', fontsize=14, fontweight='bold')
ax.set_xlabel('Passenger Class (1=First, 3=Third)')
ax.set_ylabel('Fare (£)')
plt.tight_layout()
plt.savefig('outputs/07_fare_by_class.png', dpi=150)
plt.close()
print("Saved: outputs/07_fare_by_class.png")

# ── CHART 4: Correlation Heatmap ──────────────────────────
# This is one of the most important charts in any EDA
numeric_cols = titanic.select_dtypes(include=[np.number])
correlation = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: outputs/08_correlation_heatmap.png")

print("\nAll Seaborn charts saved.")
