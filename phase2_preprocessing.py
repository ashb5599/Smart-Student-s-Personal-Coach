# ============================================================
# PHASE 2 — DATA PREPROCESSING & FEATURE ENGINEERING
# Covers: Missing values, Encoding, Scaling, Outliers,
#         Feature Engineering, Train-Test Split strategies
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# ── LOAD ────────────────────────────────────────────────────
df = pd.read_csv("StudentPerformanceFactors.csv")

# ── TARGET ──────────────────────────────────────────────────
df['Grade_Category'] = pd.cut(
    df['Exam_Score'],
    bins=[0, 60, 75, 101],
    labels=['Low', 'Medium', 'High']
)

print("=" * 60)
print("PHASE 2 — PREPROCESSING PIPELINE")
print("=" * 60)

# ============================================================
# STEP 1: MISSING VALUE TREATMENT
# ============================================================
print("\n[1] Missing Values Before Treatment:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Strategy A — Mean/Median for numeric
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  → Filled {col} with median: {median_val:.2f}")

# Strategy B — Mode for categorical
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  → Filled {col} with mode: {mode_val}")

print(f"\n  Missing after treatment: {df.isnull().sum().sum()}")

# Strategy C — KNN Imputer (advanced, shown as alternative)
# Uncomment below to use KNN imputation instead
# imputer = KNNImputer(n_neighbors=5)
# df[num_cols] = imputer.fit_transform(df[num_cols])

# ============================================================
# STEP 2: DROP DUPLICATES
# ============================================================
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[2] Duplicates removed: {before - len(df)} rows")

# ============================================================
# STEP 3: ENCODING CATEGORICAL FEATURES
# ============================================================
print("\n[3] Encoding Categorical Features...")

# Label Encoding for binary/ordinal cols
label_cols = [c for c in cat_cols if df[c].nunique() == 2]
le = LabelEncoder()
for col in label_cols:
    df[col + '_enc'] = le.fit_transform(df[col])
    print(f"  Label Encoded: {col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding for multi-class cols
ohe_cols = [c for c in cat_cols if df[c].nunique() > 2 and df[c].nunique() <= 10]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print(f"  One-Hot Encoded: {ohe_cols}")

# Drop original label cols (already encoded)
df.drop(columns=label_cols, inplace=True, errors='ignore')
# Drop Grade_Category string version
df.drop(columns=['Grade_Category'], inplace=True, errors='ignore')

print(f"\n  Shape after encoding: {df.shape}")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n[4] Feature Engineering...")

# Distraction Score (phone usage + poor sleep combined)
if 'Sleep_Hours' in df.columns:
    df['sleep_deficit'] = np.where(df['Sleep_Hours'] < 6, 1, 0)

# Study efficiency proxy
if 'Hours_Studied' in df.columns and 'Previous_Scores' in df.columns:
    df['study_efficiency'] = df['Hours_Studied'] * (df['Previous_Scores'] / 100)
    print("  → Created: study_efficiency")

# Engagement score
if 'Attendance' in df.columns and 'Hours_Studied' in df.columns:
    df['engagement_score'] = (df['Attendance'] / 100) * df['Hours_Studied']
    print("  → Created: engagement_score")

# Risk flag
if 'Attendance' in df.columns and 'Hours_Studied' in df.columns:
    df['at_risk_flag'] = ((df['Attendance'] < 60) | (df['Hours_Studied'] < 3)).astype(int)
    print("  → Created: at_risk_flag")

# ============================================================
# STEP 5: OUTLIER DETECTION & REMOVAL
# ============================================================
print("\n[5] Outlier Detection (IQR Method)...")

num_cols_now = df.select_dtypes(include=np.number).columns.tolist()
num_cols_now = [c for c in num_cols_now if c != 'Exam_Score']

outlier_counts = {}
for col in num_cols_now:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_out > 0:
        outlier_counts[col] = n_out

print(f"  Columns with outliers: {outlier_counts}")

# Clip outliers instead of dropping (preserves data)
for col in outlier_counts:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

print("  → Outliers clipped (Winsorization)")

# ============================================================
# STEP 6: PREPARE FEATURE MATRIX
# ============================================================
target_col = 'Exam_Score'
X = df.drop(columns=[target_col])
y_reg = df[target_col]                            # For regression
y_clf = pd.cut(y_reg, bins=[0,60,75,101],
               labels=[0, 1, 2]).astype(int)      # For classification

print(f"\n[6] Feature Matrix: X={X.shape}, y_reg={y_reg.shape}, y_clf={y_clf.shape}")

# ============================================================
# STEP 7: SCALING — Compare 3 Scalers
# ============================================================
print("\n[7] Scaling Comparison...")

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

scaled_versions = {}
for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X)
    scaled_versions[name] = X_scaled
    print(f"  {name}: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

# Use StandardScaler as default
X_scaled = scaled_versions['StandardScaler']

# ============================================================
# STEP 8: TRAIN-TEST SPLIT STRATEGIES
# ============================================================
print("\n[8] Train-Test Split Strategies...")

# Strategy 1 — 70/30
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_scaled, y_clf, test_size=0.30, random_state=42, stratify=y_clf)
print(f"  70/30 Split   → Train: {X_tr1.shape[0]}, Test: {X_te1.shape[0]}")

# Strategy 2 — 80/20
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_scaled, y_clf, test_size=0.20, random_state=42, stratify=y_clf)
print(f"  80/20 Split   → Train: {X_tr2.shape[0]}, Test: {X_te2.shape[0]}")

# Strategy 3 — K-Fold (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"  K-Fold (k=5)  → {kf.get_n_splits(X_scaled)} folds")

# Strategy 4 — Stratified K-Fold (k=10)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print(f"  Stratified KF (k=10) → {skf.get_n_splits(X_scaled, y_clf)} folds")

# Save processed data for next phases
import joblib
joblib.dump((X_scaled, y_reg, y_clf, X, df), 'processed_data.pkl')
joblib.dump(scalers['StandardScaler'], 'scaler.pkl')

print("\n✅ Phase 2 Complete — Preprocessing Done!")
print("   Saved: processed_data.pkl, scaler.pkl")
print("   Next: Run phase3_classical_models.py")