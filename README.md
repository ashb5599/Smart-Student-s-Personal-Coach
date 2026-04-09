# ============================================================
# README — HOW TO RUN YOUR COMPLETE ML PROJECT
# AI-Based Student Performance & Smart Guidance System
# ============================================================

STEP 0: SETUP
─────────────
  pip install -r requirements.txt

  Download dataset from Kaggle:
  https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
  → Save as "StudentPerformanceFactors.csv" in the same folder


STEP 1: RUN THE PIPELINE (in order)
────────────────────────────────────
  python phase1_eda.py                  # EDA + visualizations
  python phase2_preprocessing.py        # Preprocessing + saves processed_data.pkl
  python phase3_classical_models.py     # All ML models + comparison tables
  python phase4_clustering.py           # K-Means, Hierarchical, DBSCAN
  python phase5_deep_learning.py        # ANN training + evaluation
  python phase6_association_rules.py    # Apriori + FP-Growth + recommendations
  streamlit run phase7_streamlit_app.py # Launch the web product!


STEP 2: OUTPUTS GENERATED
──────────────────────────
  Plots (PNG):
    plot_01_score_distribution.png
    plot_02_correlation_heatmap.png
    plot_03_study_vs_score.png
    plot_04_feature_boxplots.png
    plot_05_categorical_features.png
    plot_06_confusion_matrix.png
    plot_07_feature_importance.png
    plot_08_classifier_comparison.png
    plot_09_regression_pred_vs_actual.png
    plot_10_kmeans_elbow.png
    plot_11_kmeans_clusters.png
    plot_12_dendrogram.png
    plot_13_hierarchical_clusters.png
    plot_14_dbscan_clusters.png
    plot_15_dl_training_curves.png
    plot_16_dl_confusion_matrix.png
    plot_17_dl_vs_classical.png
    plot_18_arm_scatter.png
    plot_19_arm_metrics.png

  Results (CSV):
    results_classifiers.csv         ← All classifier metrics
    results_regressors.csv          ← All regressor metrics
    results_split_strategies.csv    ← Split comparison
    results_clustering.csv          ← Clustering metrics
    results_deep_learning.csv       ← DL metrics
    results_arm_rules.csv           ← All ARM rules
    results_arm_comparison.csv      ← Apriori vs FP-Growth

  Models (PKL/Keras):
    processed_data.pkl              ← Preprocessed data
    scaler.pkl                      ← Fitted StandardScaler
    best_classifier.pkl             ← Best classification model
    best_regressor.pkl              ← Best regression model
    kmeans_model.pkl                ← Clustering model
    dl_model.keras                  ← Deep learning model
    arm_recommendations.json        ← Rule-based recommendations


STEP 3: LAUNCH PRODUCT
───────────────────────
  streamlit run phase7_streamlit_app.py
  → Opens at http://localhost:8501
  → Students enter their habits → get predictions + guidance


PROJECT STRUCTURE (Final)
──────────────────────────
  StudentPerformanceFactors.csv    ← Dataset
  requirements.txt                 ← Dependencies
  phase1_eda.py
  phase2_preprocessing.py
  phase3_classical_models.py
  phase4_clustering.py
  phase5_deep_learning.py
  phase6_association_rules.py
  phase7_streamlit_app.py          ← THE PRODUCT
  [all generated plots and CSVs]


WHAT EACH PHASE COVERS FOR VIVA/REPORT
────────────────────────────────────────
  Phase 1  → EDA, distributions, correlation analysis
  Phase 2  → Missing values, encoding, 3 scalers, feature engineering,
             outlier detection, 5 split strategies
  Phase 3  → 8 classifiers + 6 regressors, cross-validation,
             confusion matrix, feature importance, comparison table
  Phase 4  → K-Means (elbow + silhouette), Hierarchical (dendrogram),
             DBSCAN (eps tuning), 3 evaluation metrics each
  Phase 5  → ANN with dropout + batch norm, early stopping,
             loss/accuracy curves, DL vs classical comparison
  Phase 6  → Apriori + FP-Growth, support/confidence/lift/conviction,
             rule extraction, algorithm speed comparison
  Phase 7  → Full Streamlit product: grade prediction, clustering,
             distraction score, radar chart, recommendations


DATASETS USED
─────────────
  PRIMARY  : Student Performance Factors (lainguyn123 — Kaggle)
  OPTIONAL : CS Students Subject Strength (mdhossanr — Kaggle)
             Student Management Dataset (ziya07 — Kaggle)