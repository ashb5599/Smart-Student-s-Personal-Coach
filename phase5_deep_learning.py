# ============================================================
# PHASE 5 — DEEP LEARNING (ANN) MODEL
# Feedforward Neural Network for Grade Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score, log_loss)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")

# ── LOAD DATA ───────────────────────────────────────────────
X_scaled, y_reg, y_clf, X_raw, df = joblib.load('processed_data.pkl')

# One-hot encode for DL
n_classes = len(np.unique(y_clf))
y_ohe = to_categorical(y_clf, num_classes=n_classes)

X_tr, X_te, y_tr_ohe, y_te_ohe = train_test_split(
    X_scaled, y_ohe, test_size=0.2, random_state=42,
    stratify=y_clf
)
y_te_int = np.argmax(y_te_ohe, axis=1)

print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
print(f"Classes: {n_classes}")

# ============================================================
# BUILD MODEL — Feedforward ANN
# ============================================================
def build_model(input_dim, n_classes):
    model = Sequential([
        # Input + Hidden 1
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        # Hidden 2
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden 3
        Dense(64, activation='relu'),
        Dropout(0.2),

        # Hidden 4
        Dense(32, activation='relu'),

        # Output
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_tr.shape[1], n_classes)
model.summary()

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_dl_model.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
]

# ============================================================
# TRAIN
# ============================================================
print("\nTraining ANN...")
history = model.fit(
    X_tr, y_tr_ohe,
    validation_split=0.15,
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# TRAINING CURVES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ANN Training History', color='white', fontsize=13, fontweight='bold')

axes[0].plot(history.history['loss'], color='#818cf8', label='Train Loss', lw=2)
axes[0].plot(history.history['val_loss'], color='#f472b6', label='Val Loss', lw=2, linestyle='--')
axes[0].set_title('Loss Curve', color='white'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(facecolor='#1e293b', labelcolor='white')

axes[1].plot(history.history['accuracy'], color='#34d399', label='Train Acc', lw=2)
axes[1].plot(history.history['val_accuracy'], color='#fbbf24', label='Val Acc', lw=2, linestyle='--')
axes[1].set_title('Accuracy Curve', color='white'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(facecolor='#1e293b', labelcolor='white')

plt.tight_layout()
plt.savefig('plot_15_dl_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# EVALUATION
# ============================================================
y_pred_prob = model.predict(X_te)
y_pred_int  = np.argmax(y_pred_prob, axis=1)

acc  = accuracy_score(y_te_int, y_pred_int)
f1   = f1_score(y_te_int, y_pred_int, average='macro')
auc  = roc_auc_score(y_te_ohe, y_pred_prob, multi_class='ovr', average='macro')
ll   = log_loss(y_te_ohe, y_pred_prob)

print("\n" + "=" * 60)
print("DEEP LEARNING MODEL — PERFORMANCE METRICS")
print("=" * 60)
print(f"  Accuracy         : {acc:.4f}")
print(f"  F1 Score (Macro) : {f1:.4f}")
print(f"  ROC-AUC (OvR)    : {auc:.4f}")
print(f"  Log Loss         : {ll:.4f}")
print("\nClassification Report:")
print(classification_report(y_te_int, y_pred_int,
      target_names=['Low', 'Medium', 'High']))

# ── Confusion Matrix ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_te_int, y_pred_int)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Low','Med','High'], yticklabels=['Low','Med','High'])
ax.set_title('ANN Confusion Matrix', fontsize=12)
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot_16_dl_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── DL vs Best Classical Model ───────────────────────────────
print("\n" + "=" * 60)
print("DL vs BEST CLASSICAL MODEL — HEAD TO HEAD")
print("=" * 60)

try:
    best_clf = joblib.load('best_classifier.pkl')
    from sklearn.model_selection import train_test_split as tts
    X_scaled_raw, y_reg_r, y_clf_r, _, _ = joblib.load('processed_data.pkl')
    Xtr2, Xte2, ytr2, yte2 = tts(X_scaled_raw, y_clf_r, test_size=0.2, random_state=42, stratify=y_clf_r)
    yp2 = best_clf.predict(Xte2)
    clf_acc = accuracy_score(yte2, yp2)
    clf_f1  = f1_score(yte2, yp2, average='macro')

    comparison = pd.DataFrame([
        {'Model': 'Best Classical (RF/XGB)', 'Accuracy': clf_acc, 'F1_Macro': clf_f1},
        {'Model': 'ANN (Deep Learning)',     'Accuracy': acc,     'F1_Macro': f1},
    ])
    print(comparison.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    bars1 = ax.bar(x - 0.2, comparison['Accuracy'], 0.35, label='Accuracy', color='#818cf8')
    bars2 = ax.bar(x + 0.2, comparison['F1_Macro'],  0.35, label='F1 Macro', color='#34d399')
    ax.set_xticks(x); ax.set_xticklabels(comparison['Model'], rotation=10)
    ax.set_ylim(0, 1.1); ax.set_title('DL vs Classical Model', color='white', fontsize=12)
    ax.legend(facecolor='#1e293b', labelcolor='white')
    plt.tight_layout()
    plt.savefig('plot_17_dl_vs_classical.png', dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"  (Could not load classical model for comparison: {e})")

# ── SAVE ─────────────────────────────────────────────────────
model.save('dl_model.keras')
dl_metrics = {'accuracy': acc, 'f1_macro': f1, 'roc_auc': auc, 'log_loss': ll}
pd.DataFrame([dl_metrics]).to_csv('results_deep_learning.csv', index=False)

print("\n✅ Phase 5 Complete — Deep Learning Done!")
print("   Saved: dl_model.keras, results_deep_learning.csv")
print("   Next: Run phase6_association_rules.py")
