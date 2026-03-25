# matplotlib_basics.py — Day 3 of AI/ML Journey
# Topic: Matplotlib fundamentals for data visualisation

import matplotlib.pyplot as plt
import numpy as np

print("Building matplotlib charts...")

# ── CHART 1: Line Plot ─────────────────────────────────────
# Simulating a model's training loss over 10 epochs
epochs = np.arange(1, 11)
training_loss = np.array([2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 0.8, 0.75, 0.72, 0.70])
val_loss = np.array([2.6, 2.1, 1.8, 1.5, 1.3, 1.15, 1.05, 1.02, 1.0, 1.01])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, training_loss, marker='o', label='Training Loss', color='blue')
ax.plot(epochs, val_loss, marker='s', label='Validation Loss', 
        color='orange', linestyle='--')
ax.set_title('Model Training vs Validation Loss', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/01_loss_curve.png', dpi=150)
plt.close()
print("Saved: outputs/01_loss_curve.png")
# ── CHART 2: Bar Chart ─────────────────────────────────────
# Subject-wise average scores from your Day 2 data
subjects = ['Math', 'Science', 'English']
averages = [78.5, 82.3, 79.8]
colors = ['#4C72B0', '#DD8452', '#55A868']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(subjects, averages, color=colors, edgecolor='white', linewidth=0.8)

# Add value labels on top of each bar
for bar, val in zip(bars, averages):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Average Score by Subject', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score')
ax.set_ylim(70, 90)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_subject_scores.png', dpi=150)
plt.close()
print("Saved: outputs/02_subject_scores.png")
# ── CHART 3: Scatter Plot ──────────────────────────────────
# Relationship between study hours and exam scores
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 50)
exam_scores = study_hours * 7 + np.random.normal(0, 5, 50)
exam_scores = np.clip(exam_scores, 0, 100)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(study_hours, exam_scores, alpha=0.6, color='steelblue', edgecolors='white')

# Add trend line
z = np.polyfit(study_hours, exam_scores, 1)
p = np.poly1d(z)
x_line = np.linspace(1, 10, 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend line')

ax.set_title('Study Hours vs Exam Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Study Hours per Day')
ax.set_ylabel('Exam Score')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_study_vs_score.png', dpi=150)
plt.close()
print("Saved: outputs/03_study_vs_score.png")

print("\nAll 3 charts saved to outputs/")
