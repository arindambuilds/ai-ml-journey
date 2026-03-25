# titanic_eda.py — Day 3 Main Project
# Complete Exploratory Data Analysis: Titanic Dataset
# Arindam | Day 3 of AI/ML Journey
# Question we're answering: What factors determined survival on the Titanic?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set consistent style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

print("=" * 65)
print("TITANIC DATASET — EXPLORATORY DATA ANALYSIS")
print("Question: What factors determined survival?")
print("=" * 65)
# ── SECTION 1: LOAD & FIRST LOOK ──────────────────────────
print("\n[1] LOADING DATA")

df = pd.read_csv('data/titanic.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe()}")
# ── SECTION 2: MISSING VALUE ANALYSIS ─────────────────────
print("\n[2] MISSING VALUE ANALYSIS")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
}).sort_values('Missing Percentage', ascending=False)

missing_df = missing_df[missing_df['Missing Count'] > 0]
print(missing_df)

# Visualise missing values
fig, ax = plt.subplots(figsize=(8, 4))
missing_df['Missing Percentage'].plot(kind='bar', color='#E74C3C', ax=ax)
ax.set_title('Missing Data by Column (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Missing %')
ax.set_xlabel('')
plt.xticks(rotation=0)
for i, v in enumerate(missing_df['Missing Percentage']):
    ax.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/09_missing_values.png', dpi=150)
plt.close()
print("Saved: outputs/09_missing_values.png")

# Write your interpretation in a comment after running
# Which column has the most missing data?
# Can you use that column for ML? Why or why not?

# ── SECTION 3: TARGET VARIABLE ANALYSIS ───────────────────
print("\n[3] TARGET VARIABLE: SURVIVAL")

survival_counts = df['Survived'].value_counts()
survival_pct = df['Survived'].value_counts(normalize=True) * 100

print(f"Did not survive: {survival_counts[0]} ({survival_pct[0]:.1f}%)")
print(f"Survived:        {survival_counts[1]} ({survival_pct[1]:.1f}%)")
print("\nIMPORTANT: This imbalance matters when training models.")
print("A model that predicts 'no survival' for everyone gets 61%+ accuracy")
print("without learning anything. This is called the majority class baseline.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
axes[0].bar(['Did Not Survive', 'Survived'], 
            [survival_counts[0], survival_counts[1]],
            color=['#E74C3C', '#2ECC71'], edgecolor='white')
axes[0].set_title('Survival Count', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate([survival_counts[0], survival_counts[1]]):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie([survival_counts[0], survival_counts[1]],
            labels=['Did Not Survive', 'Survived'],
            colors=['#E74C3C', '#2ECC71'],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Survival Distribution', fontweight='bold')

plt.suptitle('Target Variable Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/10_survival_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/10_survival_distribution.png")
# ── SECTION 4: CATEGORICAL FEATURE ANALYSIS ───────────────
print("\n[4] CATEGORICAL FEATURES VS SURVIVAL")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Sex vs Survival
survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
axes[0].bar(survival_by_sex.index, survival_by_sex.values,
            color=['#3498DB', '#E91E63'], edgecolor='white')
axes[0].set_title('Survival Rate by Sex', fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(0, 100)
for i, v in enumerate(survival_by_sex.values):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Pclass vs Survival
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
axes[1].bar([f'Class {c}' for c in survival_by_class.index],
            survival_by_class.values,
            color=['#F39C12', '#95A5A6', '#7F8C8D'], edgecolor='white')
axes[1].set_title('Survival Rate by Passenger Class', fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(survival_by_class.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Embarked vs Survival
survival_by_embarked = df.groupby('Embarked')['Survived'].mean() * 100
axes[2].bar(survival_by_embarked.index, survival_by_embarked.values,
            color=['#1ABC9C', '#9B59B6', '#E67E22'], edgecolor='white')
axes[2].set_title('Survival Rate by Embarkation Port', fontweight='bold')
axes[2].set_ylabel('Survival Rate (%)')
axes[2].set_ylim(0, 100)
for i, v in enumerate(survival_by_embarked.values):
    axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Categorical Features vs Survival Rate', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/11_categorical_vs_survival.png', dpi=150)
plt.close()
print("Saved: outputs/11_categorical_vs_survival.png")

# Print the actual numbers
print(f"\nSurvival rate by Sex:\n{survival_by_sex.round(1)}")
print(f"\nSurvival rate by Class:\n{survival_by_class.round(1)}")
print(f"\nSurvival rate by Port:\n{survival_by_embarked.round(1)}")
# ── SECTION 5: NUMERICAL FEATURE ANALYSIS ─────────────────
print("\n[5] NUMERICAL FEATURES VS SURVIVAL")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True,
             palette={0: '#E74C3C', 1: '#2ECC71'}, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution by Survival', fontweight='bold')
axes[0, 0].legend(['Did Not Survive', 'Survived'])

# Fare distribution by survival
sns.histplot(data=df[df['Fare'] < 200], x='Fare', hue='Survived',
             bins=30, kde=True,
             palette={0: '#E74C3C', 1: '#2ECC71'}, ax=axes[0, 1])
axes[0, 1].set_title('Fare Distribution by Survival (Fare < 200)', fontweight='bold')

# Age vs Fare scatter coloured by survival
survived = df[df['Survived'] == 1]
not_survived = df[df['Survived'] == 0]
axes[1, 0].scatter(not_survived['Age'], not_survived['Fare'],
                   alpha=0.4, color='#E74C3C', label='Did Not Survive', s=20)
axes[1, 0].scatter(survived['Age'], survived['Fare'],
                   alpha=0.4, color='#2ECC71', label='Survived', s=20)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare')
axes[1, 0].set_title('Age vs Fare (coloured by Survival)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 300)

# SibSp + Parch (family size) vs survival
df['family_size'] = df['SibSp'] + df['Parch'] + 1
survival_by_family = df.groupby('family_size')['Survived'].mean() * 100
axes[1, 1].bar(survival_by_family.index, survival_by_family.values,
               color='steelblue', edgecolor='white')
axes[1, 1].set_title('Survival Rate by Family Size', fontweight='bold')
axes[1, 1].set_xlabel('Family Size (1 = alone)')
axes[1, 1].set_ylabel('Survival Rate (%)')

plt.suptitle('Numerical Features vs Survival', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/12_numerical_vs_survival.png', dpi=150)
plt.close()
print("Saved: outputs/12_numerical_vs_survival.png")# ── SECTION 6: CORRELATION ANALYSIS ───────────────────────
print("\n[6] CORRELATION ANALYSIS")

# Select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation, dtype=bool))  # hide upper triangle
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, mask=mask,
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix\n(lower triangle only)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/13_correlation_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/13_correlation_matrix.png")

# Print correlations with Survived specifically
print("\nCorrelation with Survived (sorted):")
survived_corr = correlation['Survived'].sort_values(ascending=False)
print(survived_corr.round(3))
# ── SECTION 7: KEY FINDINGS SUMMARY ───────────────────────
print("\n[7] KEY FINDINGS")
print("=" * 65)

overall_survival = df['Survived'].mean() * 100
female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
class1_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100
class3_survival = df[df['Pclass'] == 3]['Survived'].mean() * 100
alone_survival = df[df['family_size'] == 1]['Survived'].mean() * 100
family_survival = df[df['family_size'].between(2, 4)]['Survived'].mean() * 100

print(f"Overall survival rate:           {overall_survival:.1f}%")
print(f"Female survival rate:            {female_survival:.1f}%")
print(f"Male survival rate:              {male_survival:.1f}%")
print(f"First class survival rate:       {class1_survival:.1f}%")
print(f"Third class survival rate:       {class3_survival:.1f}%")
print(f"Travelling alone survival rate:  {alone_survival:.1f}%")
print(f"Small family (2-4) survival:     {family_survival:.1f}%")

print("\n--- CONCLUSIONS ---")
print("1. Sex is the strongest predictor of survival")
print("2. Passenger class has significant impact — wealth = survival advantage")
print("3. Small family groups had better survival than solo travellers")
print("4. Age alone is a weak predictor but children had higher survival rates")
print("5. Fare correlates with survival but is largely explained by class")

print("\n--- ML READINESS ---")
print("Features to USE in a model:  Sex, Pclass, Age, Fare, family_size")
print("Features to DROP:            Cabin (too many missing), Name, Ticket")
print("Features needing work:       Age (22% missing — needs imputation)")
print("Target variable:             Survived (binary classification problem)")

print("\n" + "=" * 65)
print("EDA Complete. Dataset is understood. Ready for modelling.")
print("=" * 65)