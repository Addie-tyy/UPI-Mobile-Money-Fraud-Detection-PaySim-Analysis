# UPI/Mobile Money Fraud Detection — PaySim Analysis

SQL-driven exploration, leakage-aware feature engineering, explainable ML, and a precision/recall tradeoff framed as a business decision — simulating the kind of risk-analyst work done at a UPI-scale payments company.

**Dataset:** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle) — 6.3M+ real transactions, standard academic mobile-money fraud benchmark.

# Core Tech Stack
Data Engineering & Querying: SQLite3, Pandas, NumPy

Machine Learning: Scikit-Learn, LightGBM / XGBoost

Explainability: SHAP (TreeExplainer)

Environment: Python 3.10+, Jupyter / Google Colab



End-to-End Methodology
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



Honest Limitations & Production Considerations
Synthetic Nature: PaySim does not reflect modern UPI features (device fingerprinting, VPA aliases, SIM binding, or tokenized flows).

Absence of Temporal Splits: The pipeline uses stratified random sampling. A live production engine requires out-of-time (OOT) validation splits to track adversarial drift.

Financial Sizing: Dollar/Rupee fraud-loss figures represent sample accounting derived from test set totals to demonstrate metric conversion, not empirical real-world losses.
