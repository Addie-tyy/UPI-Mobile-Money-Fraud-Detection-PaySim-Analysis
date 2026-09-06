# UPI/Mobile Money Fraud Detection — PaySim Analysis

SQL-driven exploration, leakage-aware feature engineering, explainable ML, and a precision/recall tradeoff framed as a business decision — simulating the kind of risk-analyst work done at a UPI-scale payments company.

**Dataset:** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle) — 6.3M+ real transactions, standard academic mobile-money fraud benchmark.

# Core Tech Stack
Data Engineering & Querying: SQLite3, Pandas, NumPy

Machine Learning: Scikit-Learn, LightGBM / XGBoost

Explainability: SHAP (TreeExplainer)

Environment: Python 3.10+, Jupyter / Google Colab

## End-to-End Methodology

```
6.3M Raw Transactions (CSV)
            │
            ▼
┌───────────────────────────────┐
│ 1. Exploratory SQL Layer      │ ──► Isolate TRANSFER & CASH_OUT
│    (SQLite / CTEs / RANK)     │ ──► Reject single-account hypothesis
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 2. Feature Engineering        │ ──► Engineer balance delta signals
│    & Leakage Discovery        │ ──► Identify 'full_drain' synthetic artifact
└──────────────┬────────────────┘
               │
       ┌───────┴────────────────┐
       ▼                        ▼
┌──────────────┐         ┌──────────────┐
│ Model A      │         │ Model B      │
│ (With Leak)  │         │ (Leak-Free)  │
│ Recall: 99.7%│         │ Recall: 86.6%│
└──────────────┘         └──────┬───────┘
                                │
                                ▼
┌───────────────────────────────┐
│ 3. Model Explainability       │ ──► TreeExplainer SHAP
│    & Interpretability         │ ──► Reconstructed delta importance
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 4. Threshold Optimization     │ ──► Precision target: 95.0%
│    & Business Calibration     │ ──► Optimal cutoff: 0.864
└───────────────────────────────┘
## Executive Summary & Business Impact

| Metric / Decision Variable | Baseline Rule (`isFlaggedFraud`) | Honest Production Model (Model B @ 0.86 Threshold) | Net Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Fraud Detection Rate (Recall)** | 0.19% (16 / 8,213) | **71.0%** (154 / 217 in Test) | **~370x improvement** |
| **Operational Precision** | ~100% | **95.0%** | 1 false alert per 20 flags |
| **False Positive Rate (FPR)** | ~0.000% | **0.011%** (8 false alerts in 74,783) | Minimal manual review bloat |
| **Fraud Prevented (Test Set)** | ~₹0.23 Cr | **~₹22.6 Cr** (Avg ₹14.68L / fraud) | **+₹22.37 Cr recovered** |
| **Remaining Loss Exposure** | ~₹120.3 Cr | **~₹9.2 Cr** (63 missed fraud cases) | Drastic risk mitigation |



Honest Limitations & Production Considerations
Synthetic Nature: PaySim does not reflect modern UPI features (device fingerprinting, VPA aliases, SIM binding, or tokenized flows).

Absence of Temporal Splits: The pipeline uses stratified random sampling. A live production engine requires out-of-time (OOT) validation splits to track adversarial drift.

Financial Sizing: Dollar/Rupee fraud-loss figures represent sample accounting derived from test set totals to demonstrate metric conversion, not empirical real-world losses.
