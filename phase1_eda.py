# ============================================================
# PHASE 1 — DATA LOADING, MERGING & EXPLORATORY DATA ANALYSIS
# Project: AI-Based Student Performance & Smart Guidance System
# Dataset: Student Performance Factors (lainguyn123 - Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ───────────────────────────────────────────────────
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.facecolor'] = '#0f1117'
plt.rcParams['figure.facecolor'] = '#0f1117'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
sns.set_palette("husl")

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Download from: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
df = pd.read_csv("StudentPerformanceFactors.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape        : {df.shape}")
print(f"Rows         : {df.shape[0]}")
print(f"Columns      : {df.shape[1]}")
print(f"\nColumn Names :\n{list(df.columns)}")
print(f"\nData Types :\n{df.dtypes}")
print(f"\nMissing Values :\n{df.isnull().sum()}")
print(f"\nDuplicates   : {df.duplicated().sum()}")

# ============================================================
# 2. CREATE TARGET VARIABLE — Grade Category
# ============================================================
# Bin Exam_Score into 3 classes for classification
df['Grade_Category'] = pd.cut(
    df['Exam_Score'],
    bins=[0, 60, 75, 101],
    labels=['Low', 'Medium', 'High']
)
print(f"\nGrade Distribution:\n{df['Grade_Category'].value_counts()}")

# ============================================================
# 3. BASIC STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().round(2).to_string())

# ============================================================
# 4. EDA — VISUALIZATIONS
# ============================================================

# 4a. Exam Score Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Exam Score Distribution', color='white', fontsize=14, fontweight='bold')

axes[0].hist(df['Exam_Score'], bins=30, color='#818cf8', edgecolor='#312e81', alpha=0.85)
axes[0].set_title('Histogram', color='white')
axes[0].set_xlabel('Exam Score')
axes[0].set_ylabel('Count')

df['Grade_Category'].value_counts().plot(kind='bar', ax=axes[1], color=['#ef4444','#f59e0b','#22c55e'], edgecolor='black')
axes[1].set_title('Grade Category Distribution', color='white')
axes[1].set_xlabel('Grade')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('plot_01_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot_01_score_distribution.png")

# 4b. Correlation Heatmap (numerical only)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
fig, ax = plt.subplots(figsize=(12, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title('Feature Correlation Heatmap', color='white', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('plot_02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot_02_correlation_heatmap.png")

# 4c. Study Hours vs Exam Score by Grade
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'Low': '#ef4444', 'Medium': '#f59e0b', 'High': '#22c55e'}
for grade, grp in df.groupby('Grade_Category'):
    ax.scatter(grp['Hours_Studied'], grp['Exam_Score'],
               label=grade, color=colors[grade], alpha=0.5, s=20)
ax.set_xlabel('Hours Studied per Week')
ax.set_ylabel('Exam Score')
ax.set_title('Study Hours vs Exam Score by Grade', color='white', fontsize=12)
ax.legend(title='Grade', facecolor='#1e293b', labelcolor='white')
plt.tight_layout()
plt.savefig('plot_03_study_vs_score.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot_03_study_vs_score.png")

# 4d. Key Features Boxplot by Grade
key_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
available = [f for f in key_features if f in df.columns]

fig, axes = plt.subplots(1, len(available), figsize=(16, 5))
fig.suptitle('Key Feature Distribution by Grade Category', color='white', fontsize=13, fontweight='bold')
palette = {'Low': '#ef4444', 'Medium': '#f59e0b', 'High': '#22c55e'}

for i, feat in enumerate(available):
    sns.boxplot(data=df, x='Grade_Category', y=feat, ax=axes[i],
                order=['Low','Medium','High'], palette=palette)
    axes[i].set_title(feat.replace('_',' '), color='white', fontsize=10)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.savefig('plot_04_feature_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot_04_feature_boxplots.png")

# 4e. Categorical columns — count plots
cat_cols = df.select_dtypes(include='object').columns.drop('Grade_Category', errors='ignore').tolist()
cat_cols = [c for c in cat_cols if df[c].nunique() <= 10][:6]  # max 6

if cat_cols:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle('Categorical Feature Distributions', color='white', fontsize=13)
    for i, col in enumerate(cat_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, hue='Grade_Category',
                      ax=axes[i], order=order, palette=palette,
                      hue_order=['Low','Medium','High'])
        axes[i].set_title(col.replace('_',' '), color='white', fontsize=9)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].legend(fontsize=7, facecolor='#1e293b', labelcolor='white')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig('plot_05_categorical_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_05_categorical_features.png")

print("\n✅ Phase 1 Complete — EDA Done!")
print("Next: Run phase2_preprocessing.py")
