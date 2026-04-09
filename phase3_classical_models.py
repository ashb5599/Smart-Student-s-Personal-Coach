# ============================================================
# PHASE 3 — CLASSICAL ML MODELS
# Covers: Regression + Classification models, all split
#         strategies, full performance comparison table
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score)

# ── CLASSIFIERS ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

# ── LOAD DATA ───────────────────────────────────────────────
X_scaled, y_reg, y_clf, X_raw, df = joblib.load('processed_data.pkl')
print(f"Loaded: X={X_scaled.shape}, y_clf classes={np.unique(y_clf)}")

# ── SPLITS ──────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_clf, test_size=0.2,
                                            random_state=42, stratify=y_clf)
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_scaled, y_reg, test_size=0.2,
                                                     random_state=42)

print("=" * 60)
print("SECTION A — CLASSIFICATION MODELS")
print("=" * 60)

# ── DEFINE ALL CLASSIFIERS ───────────────────────────────────
classifiers = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)':              KNeighborsClassifier(n_neighbors=5),
    'KNN (k=11)':             KNeighborsClassifier(n_neighbors=11),
    'Decision Tree':          DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)':              SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes':            GaussianNB(),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
}
if HAS_XGB:
    classifiers['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42,
                                            eval_metric='mlogloss', use_label_encoder=False)

# ── TRAIN & EVALUATE ─────────────────────────────────────────
clf_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV(5)':>8}")
print("-" * 70)

best_clf = None
best_f1 = 0

for name, model in classifiers.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)

    try:
        y_prob = model.predict_proba(X_te)
        auc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='macro')
    except:
        auc = float('nan')

    cv_scores = cross_val_score(model, X_scaled, y_clf, cv=skf, scoring='f1_macro')
    cv_mean = cv_scores.mean()

    clf_results.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1_Score': f1, 'ROC_AUC': auc, 'CV_F1(5Fold)': cv_mean
    })

    print(f"{name:<25} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc:>7.4f} {cv_mean:>8.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_clf = (name, model)

clf_df = pd.DataFrame(clf_results).sort_values('F1_Score', ascending=False)
print(f"\n🏆 Best Classifier: {best_clf[0]} (F1={best_f1:.4f})")

# ── CONFUSION MATRIX (Best Model) ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
y_pred_best = best_clf[1].predict(X_te)
cm = confusion_matrix(y_te, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Low','Med','High'], yticklabels=['Low','Med','High'])
ax.set_title(f'Confusion Matrix — {best_clf[0]}', fontsize=12)
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot_06_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── FEATURE IMPORTANCE (Random Forest) ──────────────────────
rf = classifiers['Random Forest']
feature_names = [f'F{i}' for i in range(X_scaled.shape[1])]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)[:15]

fig, ax = plt.subplots(figsize=(10, 5))
importances.plot(kind='bar', ax=ax, color='#818cf8', edgecolor='#312e81')
ax.set_title('Top 15 Feature Importances (Random Forest)', color='white', fontsize=12)
ax.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('plot_07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── SPLIT STRATEGY COMPARISON ────────────────────────────────
print("\n" + "=" * 60)
print("SPLIT STRATEGY COMPARISON (Random Forest)")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
split_results = []

for ratio in [(0.3, '70/30'), (0.2, '80/20'), (0.15, '85/15')]:
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_clf, test_size=ratio[0],
                                            stratify=y_clf, random_state=42)
    rf_model.fit(Xtr, ytr)
    yp = rf_model.predict(Xte)
    f1 = f1_score(yte, yp, average='macro')
    acc = accuracy_score(yte, yp)
    split_results.append({'Split': ratio[1], 'Accuracy': acc, 'F1_Score': f1,
                           'Train_Size': len(Xtr), 'Test_Size': len(Xte)})

for k in [5, 10]:
    skf_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv = cross_val_score(rf_model, X_scaled, y_clf, cv=skf_k, scoring='f1_macro')
    split_results.append({'Split': f'SKFold-{k}', 'Accuracy': np.nan,
                           'F1_Score': cv.mean(), 'Train_Size': '-', 'Test_Size': '-'})

split_df = pd.DataFrame(split_results)
print(split_df.to_string(index=False))

# ── MODEL COMPARISON PLOT ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Classifier Performance Comparison', color='white', fontsize=13, fontweight='bold')

clf_df_plot = clf_df.set_index('Model')
clf_df_plot[['Accuracy','F1_Score','ROC_AUC']].plot(kind='bar', ax=axes[0],
    color=['#818cf8','#34d399','#f472b6'], edgecolor='black')
axes[0].set_title('Accuracy / F1 / AUC', color='white')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(facecolor='#1e293b', labelcolor='white')
axes[0].set_ylim(0, 1.1)

clf_df_plot['CV_F1(5Fold)'].plot(kind='bar', ax=axes[1], color='#fbbf24', edgecolor='black')
axes[1].set_title('Cross-Validation F1 (5-Fold)', color='white')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('plot_08_classifier_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SECTION B — REGRESSION MODELS
# ============================================================
print("\n" + "=" * 60)
print("SECTION B — REGRESSION MODELS (Predict Exam Score)")
print("=" * 60)

regressors = {
    'Linear Regression':          LinearRegression(),
    'Ridge Regression':           Ridge(alpha=1.0),
    'Lasso Regression':           Lasso(alpha=0.1),
    'KNN Regressor (k=5)':        KNeighborsRegressor(n_neighbors=5),
    'Decision Tree Regressor':    DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest Regressor':    RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regr.':    GradientBoostingRegressor(n_estimators=100, random_state=42) if hasattr(__import__('sklearn.ensemble', fromlist=['GradientBoostingRegressor']), 'GradientBoostingRegressor') else None,
}
try:
    from sklearn.ensemble import GradientBoostingRegressor
    regressors['Gradient Boosting Regr.'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
except:
    pass

reg_results = []
print(f"\n{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 58)

best_reg = None
best_r2 = -999

for name, model in regressors.items():
    if model is None: continue
    model.fit(X_tr_r, y_tr_r)
    y_pred_r = model.predict(X_te_r)
    mae  = mean_absolute_error(y_te_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
    r2   = r2_score(y_te_r, y_pred_r)
    reg_results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"{name:<30} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_reg = (name, model)

reg_df = pd.DataFrame(reg_results).sort_values('R2', ascending=False)
print(f"\n🏆 Best Regressor: {best_reg[0]} (R²={best_r2:.4f})")

# ── Predicted vs Actual Plot ─────────────────────────────────
y_pred_best_r = best_reg[1].predict(X_te_r)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_te_r, y_pred_best_r, alpha=0.4, color='#818cf8', s=20)
ax.plot([y_te_r.min(), y_te_r.max()], [y_te_r.min(), y_te_r.max()], 'r--', lw=2)
ax.set_xlabel('Actual Score'); ax.set_ylabel('Predicted Score')
ax.set_title(f'Predicted vs Actual — {best_reg[0]}', color='white')
plt.tight_layout()
plt.savefig('plot_09_regression_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()

# ── SAVE RESULTS & BEST MODELS ───────────────────────────────
clf_df.to_csv('results_classifiers.csv', index=False)
reg_df.to_csv('results_regressors.csv', index=False)
split_df.to_csv('results_split_strategies.csv', index=False)
joblib.dump(best_clf[1], 'best_classifier.pkl')
joblib.dump(best_reg[1], 'best_regressor.pkl')

print("\n✅ Phase 3 Complete — Classical ML Done!")
print("   Saved: results_classifiers.csv, results_regressors.csv")
print("   Saved: best_classifier.pkl, best_regressor.pkl")
print("   Next: Run phase4_clustering.py")
