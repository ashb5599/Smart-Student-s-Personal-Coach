# ============================================================
# PHASE 6 — ASSOCIATION RULE MINING
# Apriori + FP-Growth, Support/Confidence/Lift metrics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# pip install mlxtend
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── LOAD RAW DATA ────────────────────────────────────────────
df_raw = pd.read_csv("StudentPerformanceFactors.csv")

print("=" * 60)
print("PHASE 6 — ASSOCIATION RULE MINING")
print("=" * 60)

# ============================================================
# STEP 1 — BINNING (Discretize numeric into categories)
# ============================================================
print("\n[1] Discretizing numerical features for ARM...")

df_arm = pd.DataFrame()

# Study Hours → Low / Medium / High
if 'Hours_Studied' in df_raw.columns:
    df_arm['StudyHours'] = pd.cut(df_raw['Hours_Studied'],
        bins=[0, 10, 20, 50], labels=['LowStudy', 'MedStudy', 'HighStudy'])

# Attendance → Poor / Average / Good
if 'Attendance' in df_raw.columns:
    df_arm['Attendance'] = pd.cut(df_raw['Attendance'],
        bins=[0, 60, 80, 101], labels=['PoorAttend', 'AvgAttend', 'GoodAttend'])

# Sleep Hours → Low / Normal / High
if 'Sleep_Hours' in df_raw.columns:
    df_arm['Sleep'] = pd.cut(df_raw['Sleep_Hours'],
        bins=[0, 5, 7, 12], labels=['LowSleep', 'NormalSleep', 'HighSleep'])

# Previous Scores → Low / Medium / High
if 'Previous_Scores' in df_raw.columns:
    df_arm['PrevScore'] = pd.cut(df_raw['Previous_Scores'],
        bins=[0, 50, 75, 101], labels=['LowPrev', 'MedPrev', 'HighPrev'])

# Tutoring → Yes/No
if 'Tutoring_Sessions' in df_raw.columns:
    df_arm['Tutoring'] = pd.cut(df_raw['Tutoring_Sessions'],
        bins=[-1, 0, 2, 100], labels=['NoTutor', 'FewTutor', 'ActiveTutor'])

# Target Grade
df_arm['Grade'] = pd.cut(df_raw['Exam_Score'],
    bins=[0, 60, 75, 101], labels=['GradeLow', 'GradeMed', 'GradeHigh'])

# Pass/Fail binary
df_arm['PassFail'] = np.where(df_raw['Exam_Score'] >= 60, 'Pass', 'Fail')

# Parental involvement if available
for col in ['Parental_Involvement', 'Internet_Access', 'Extracurricular_Activities']:
    if col in df_raw.columns:
        df_arm[col] = df_raw[col].astype(str).str.strip()

df_arm.dropna(inplace=True)
print(f"  ARM dataset shape: {df_arm.shape}")
print(f"  Columns: {list(df_arm.columns)}")

# ============================================================
# STEP 2 — CONVERT TO ONE-HOT (Basket format)
# ============================================================
print("\n[2] Converting to transaction/basket format...")

# Convert each row to list of active items
transactions = []
for _, row in df_arm.iterrows():
    transaction = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
    transactions.append(transaction)

# Encode
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
print(f"  Transaction matrix: {df_encoded.shape}")

# ============================================================
# STEP 3 — APRIORI ALGORITHM
# ============================================================
print("\n[3] Running Apriori Algorithm...")
t0 = time.time()
freq_itemsets_apriori = apriori(df_encoded, min_support=0.15, use_colnames=True, max_len=4)
t_apriori = time.time() - t0

rules_apriori = association_rules(freq_itemsets_apriori, metric='lift', min_threshold=1.0)
rules_apriori['conviction'] = (1 - rules_apriori['consequent support']) / (1 - rules_apriori['confidence'] + 1e-9)
rules_apriori['leverage'] = rules_apriori['support'] - (rules_apriori['antecedent support'] * rules_apriori['consequent support'])

print(f"  Time taken      : {t_apriori:.2f}s")
print(f"  Frequent items  : {len(freq_itemsets_apriori)}")
print(f"  Rules generated : {len(rules_apriori)}")

# ============================================================
# STEP 4 — FP-GROWTH ALGORITHM
# ============================================================
print("\n[4] Running FP-Growth Algorithm...")
t0 = time.time()
freq_itemsets_fp = fpgrowth(df_encoded, min_support=0.15, use_colnames=True, max_len=4)
t_fpgrowth = time.time() - t0

rules_fp = association_rules(freq_itemsets_fp, metric='lift', min_threshold=1.0)
print(f"  Time taken      : {t_fpgrowth:.2f}s")
print(f"  Frequent items  : {len(freq_itemsets_fp)}")
print(f"  Rules generated : {len(rules_fp)}")

# ── Speed comparison ─────────────────────────────────────────
print(f"\n  ⚡ Speed Comparison: Apriori={t_apriori:.3f}s  |  FP-Growth={t_fpgrowth:.3f}s")
print(f"  FP-Growth is {t_apriori/t_fpgrowth:.1f}x faster" if t_fpgrowth > 0 else "")

# ============================================================
# STEP 5 — TOP RULES ANALYSIS
# ============================================================
print("\n[5] Top Rules by Lift (Apriori):")
print("=" * 60)

top_rules = rules_apriori.sort_values('lift', ascending=False).head(20)
top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))

for _, row in top_rules.head(10).iterrows():
    print(f"\n  IF   {row['antecedents_str']}")
    print(f"  THEN {row['consequents_str']}")
    print(f"       Supp={row['support']:.3f}  Conf={row['confidence']:.3f}  Lift={row['lift']:.3f}")

# Filter rules that predict grade
grade_rules = rules_apriori[
    rules_apriori['consequents'].apply(lambda x: any('Grade' in str(i) or 'Pass' in str(i) or 'Fail' in str(i) for i in x))
].sort_values('lift', ascending=False)

print(f"\n  Rules predicting Grade/Pass: {len(grade_rules)}")
print("\n  TOP 5 GRADE-PREDICTING RULES:")
for _, row in grade_rules.head(5).iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"    {ant} → {con}  [Lift={row['lift']:.2f}, Conf={row['confidence']:.2f}]")

# ============================================================
# STEP 6 — VISUALIZATIONS
# ============================================================

# 6a. Support vs Confidence scatter
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(rules_apriori['support'], rules_apriori['confidence'],
                c=rules_apriori['lift'], cmap='plasma', alpha=0.6, s=30)
plt.colorbar(sc, ax=ax, label='Lift')
ax.set_xlabel('Support'); ax.set_ylabel('Confidence')
ax.set_title('Association Rules: Support vs Confidence (color=Lift)', color='white', fontsize=12)
plt.tight_layout()
plt.savefig('plot_18_arm_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# 6b. Top rules by different metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Top 10 Rules by Metric', color='white', fontsize=12)
metrics = ['lift', 'confidence', 'support']
colors = ['#818cf8', '#34d399', '#fbbf24']
for i, (metric, color) in enumerate(zip(metrics, colors)):
    top10 = rules_apriori.nlargest(10, metric)
    labels = [f"R{j+1}" for j in range(len(top10))]
    axes[i].barh(labels, top10[metric], color=color, edgecolor='black')
    axes[i].set_title(metric.capitalize(), color='white')
    axes[i].set_xlabel(metric)
plt.tight_layout()
plt.savefig('plot_19_arm_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

# 6c. Algorithm comparison
algo_compare = pd.DataFrame([
    {'Algorithm': 'Apriori',   'Time(s)': t_apriori, 'Frequent Itemsets': len(freq_itemsets_apriori), 'Rules': len(rules_apriori)},
    {'Algorithm': 'FP-Growth', 'Time(s)': t_fpgrowth, 'Frequent Itemsets': len(freq_itemsets_fp),     'Rules': len(rules_fp)},
])
print("\n" + "=" * 60)
print("ARM ALGORITHM COMPARISON")
print("=" * 60)
print(algo_compare.to_string(index=False))

# ============================================================
# STEP 7 — RECOMMENDATION RULE EXTRACTION (for Product)
# ============================================================
print("\n[7] Extracting Recommendation Rules for Product...")

# Rules that lead to GradeHigh
improvement_rules = rules_apriori[
    rules_apriori['consequents'].apply(lambda x: 'Grade=GradeHigh' in x or 'PassFail=Pass' in x)
].sort_values('confidence', ascending=False)

recommendations = []
for _, row in improvement_rules.head(15).iterrows():
    ant = list(row['antecedents'])
    rec = {
        'if_student_has': ant,
        'then_likely': list(row['consequents']),
        'confidence': row['confidence'],
        'lift': row['lift'],
        'support': row['support']
    }
    recommendations.append(rec)

import json
with open('arm_recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print(f"  {len(recommendations)} recommendation rules saved to arm_recommendations.json")

# ── SAVE ─────────────────────────────────────────────────────
rules_apriori.to_csv('results_arm_rules.csv', index=False)
algo_compare.to_csv('results_arm_comparison.csv', index=False)

print("\n✅ Phase 6 Complete — Association Rule Mining Done!")
print("   Saved: results_arm_rules.csv, arm_recommendations.json")
print("   Next: Run phase7_streamlit_app.py")
