# student_analyzer.py — Day 2 Mini Project
# Student Performance Analyzer using NumPy + Pandas
# Arindam | Day 2 of AI/ML Journey

import pandas as pd
import numpy as np

print("=" * 60)
print("STUDENT PERFORMANCE ANALYZER")
print("=" * 60)

# ── STEP 1: LOAD DATA ──────────────────────────────────────
df = pd.read_csv('data/students.csv')

print("\n[1] RAW DATA OVERVIEW")
print(f"Total students: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nFirst look:\n{df.head()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
# ── STEP 2: CLEAN DATA ─────────────────────────────────────
print("\n[2] CLEANING DATA")

# Store original missing count for reporting
missing_before = df.isnull().sum().sum()

# Fill missing subject scores with column median
# Why median? More robust to outliers than mean
for col in ['math', 'science', 'english']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Filled missing {col} with median: {median_val:.1f}")

# Fill missing attendance with mean
mean_attendance = df['attendance'].mean()
df['attendance'] = df['attendance'].fillna(round(mean_attendance, 1))
print(f"  Filled missing attendance with mean: {mean_attendance:.1f}")

missing_after = df.isnull().sum().sum()
print(f"\nMissing values: {missing_before} → {missing_after}")
print("Data is clean. No missing values remain.")
# ── STEP 3: TRANSFORM DATA ─────────────────────────────────
print("\n[3] TRANSFORMATIONS")

# Calculate total and average score per student
df['total_score'] = df['math'] + df['science'] + df['english']
df['average_score'] = (df['total_score'] / 3).round(2)

# Grade assignment using NumPy's where (industry standard approach)
# This is more efficient than a Python for loop
conditions = [
    df['average_score'] >= 90,
    df['average_score'] >= 80,
    df['average_score'] >= 70,
    df['average_score'] >= 60,
]
grades = ['A', 'B', 'C', 'D']
df['grade'] = np.select(conditions, grades, default='F')

# Attendance status
df['attendance_status'] = np.where(df['attendance'] >= 75, 'Regular', 'At Risk')

print("Transformed data sample:")
print(df[['name', 'math', 'science', 'english', 'average_score', 
          'grade', 'attendance_status']].to_string(index=False))
# ── STEP 4: ANALYSIS ───────────────────────────────────────
print("\n[4] INSIGHTS")

# Overall statistics
print("--- Overall Performance ---")
print(f"Class average: {df['average_score'].mean():.2f}")
print(f"Highest scorer: {df.loc[df['average_score'].idxmax(), 'name']} "
      f"({df['average_score'].max():.2f})")
print(f"Lowest scorer: {df.loc[df['average_score'].idxmin(), 'name']} "
      f"({df['average_score'].min():.2f})")

# Grade distribution
print("\n--- Grade Distribution ---")
grade_counts = df['grade'].value_counts().sort_index()
for grade, count in grade_counts.items():
    bar = '█' * count
    print(f"  Grade {grade}: {bar} ({count} students)")

# City-wise performance
print("\n--- Average Score by City ---")
city_performance = df.groupby('city')['average_score'].mean().sort_values(
    ascending=False).round(2)
for city, avg in city_performance.items():
    print(f"  {city}: {avg}")

# At-risk students
at_risk = df[df['attendance_status'] == 'At Risk']
print(f"\n--- At-Risk Students (attendance < 75%) ---")
if len(at_risk) > 0:
    print(at_risk[['name', 'attendance', 'average_score', 'grade']].to_string(
        index=False))
else:
    print("  None")

# Students who failed previously but improved
improved = df[(df['passed_prev_year'] == 'No') & (df['average_score'] >= 70)]
print(f"\n--- Previously Failed but Improved (avg ≥ 70) ---")
if len(improved) > 0:
    print(improved[['name', 'average_score', 'grade']].to_string(index=False))
else:
    print("  None")
    # ── STEP 5: EXPORT CLEAN DATA ──────────────────────────────
print("\n[5] SAVING RESULTS")

# Save clean + transformed data
df.to_csv('data/students_analyzed.csv', index=False)
print("Saved: data/students_analyzed.csv")

# Save summary statistics
summary = df[['math', 'science', 'english', 'average_score', 
              'attendance']].describe().round(2)
summary.to_csv('data/summary_stats.csv')
print("Saved: data/summary_stats.csv")

print("\n" + "=" * 60)
print("Analysis complete.")
print("=" * 60)