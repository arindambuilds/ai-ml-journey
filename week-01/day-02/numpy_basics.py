# numpy_basics.py - Day 2 of AI/ML Journey
# Numpy = Numerical Python. Every AI/ML library is built on top of it.

import numpy as np

# ── PART 1: Creating arrays ───────────────────────────────────

# A NumPy array is like a Python list but 100x faster for math
a = np.array([1, 2, 3, 4, 5])
print("Basic array:", a)
print("Type:", type(a))
print("Shape:", a.shape)       # (5,) means 5 elements, 1 dimension
print("Data type:", a.dtype)   # int64 — what kind of numbers

# 2D array — think of it as a table (rows x columns)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\n2D Matrix:\n", matrix)
print("Shape:", matrix.shape)  # (3, 3) = 3 rows, 3 columns

# ── PART 2: Creating arrays automatically ────────────────────

zeros = np.zeros((3, 4))       # 3 rows, 4 cols of zeros
ones = np.ones((2, 3))         # 2 rows, 3 cols of ones
r = np.random.rand(3, 3)       # random numbers between 0 and 1
rng = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8] — like range()
lin = np.linspace(0, 1, 5)     # 5 evenly spaced numbers from 0 to 1

print("\nZeros:\n", zeros)
print("Ones:\n", ones)
print("Random:\n", r)
print("Arange:", rng)
print("Linspace:", lin)

# ── PART 3: Array math — this is why NumPy exists ────────────

x = np.array([10, 20, 30, 40, 50])
y = np.array([1, 2, 3, 4, 5])

print("\nAddition:", x + y)        # [11, 22, 33, 44, 55]
print("Multiplication:", x * y)    # [10, 40, 90, 160, 250]
print("Division:", x / y)          # [10, 10, 10, 10, 10]
print("Square root:", np.sqrt(x))  # sqrt of each element
print("Mean:", np.mean(x))         # average
print("Sum:", np.sum(x))           # total
print("Max:", np.max(x))           # largest
print("Min:", np.min(x))           # smallest
print("Std dev:", np.std(x))       # standard deviation

# ── PART 4: Indexing and slicing ─────────────────────────────

m = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

print("\nFull matrix:\n", m)
print("Row 0:", m[0])           # first row
print("Element [1][2]:", m[1][2])  # row 1, col 2 = 60
print("First 2 rows:\n", m[:2])    # rows 0 and 1
print("Last column:", m[:, -1])    # all rows, last column

# ── PART 5: Why this matters for AI/ML ───────────────────────
# In ML, your dataset is a NumPy array.
# 1000 images of 28x28 pixels = array of shape (1000, 784)
# Model weights are arrays. Predictions are arrays.
# Everything is array math. This is the foundation.

print("\n--- Day 2 NumPy: Complete ---")

# numpy_basics.py — Day 2 of AI/ML Journey
# Topic: NumPy fundamentals for ML

import numpy as np

# 1. Creating arrays
scores = np.array([85, 92, 78, 95, 88, 76, 91, 83])
print("Scores array:", scores)
print("Shape:", scores.shape)
print("Data type:", scores.dtype)
print("Number of dimensions:", scores.ndim)

# 2. Basic operations — these work element-by-element
print("\n--- Array Operations ---")
print("Mean score:", np.mean(scores))
print("Max score:", np.max(scores))
print("Min score:", np.min(scores))
print("Standard deviation:", np.std(scores))
print("Scores above 85:", scores[scores > 85])

# 3. 2D arrays — this is what your data looks like in ML
student_data = np.array([
    [85, 92, 78],   # student 1: math, science, english
    [95, 88, 91],   # student 2
    [76, 83, 70],   # student 3
    [91, 95, 88]    # student 4
])
print("\n--- 2D Array ---")
print("Shape:", student_data.shape)         # (4, 3) — 4 students, 3 subjects
print("All math scores:", student_data[:, 0])     # first column
print("Student 1 scores:", student_data[0, :])    # first row
print("Average per student:", np.mean(student_data, axis=1))
print("Average per subject:", np.mean(student_data, axis=0))
