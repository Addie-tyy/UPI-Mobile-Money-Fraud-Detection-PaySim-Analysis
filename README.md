# UPI/Mobile Money Fraud Detection — PaySim Analysis

SQL-driven exploration, leakage-aware feature engineering, explainable ML, and a precision/recall tradeoff framed as a business decision — simulating the kind of risk-analyst work done at a UPI-scale payments company.

**Dataset:** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle) — 6.3M+ real transactions, standard academic mobile-money fraud benchmark.

## Core Tech Stack
*Data Engineering & Querying:* SQLite3, Pandas, NumPy

*Machine Learning:* Scikit-Learn, LightGBM / XGBoost

*Explainability:* SHAP (TreeExplainer)

*Environment:* Python 3.10+, Jupyter / Google Colab

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
```

### 1. Exploratory SQL (SQLite / In-Memory DB)
* **Transaction Isolation**: Confirmed fraud occurs exclusively in `TRANSFER` and `CASH_OUT` vectors; filtered out non-vulnerable types (`PAYMENT`, `CASH_IN`, `DEBIT`) early to reduce pipeline memory overhead.
* **Account Dynamics**: Executed window functions (`RANK()`, `ROW_NUMBER()`) and CTEs to analyze sender activity profiles. Disproved the "throwaway account" heuristic—both fraudulent and legitimate sender IDs averaged ~1.0 transactions/account due to the dataset's synthetic generation parameters.

### 2. Feature Engineering & The Leakage Trap
Two balance-consistency signals were discovered:
* `full_drain`: Flagging transactions where the sender's entire originating balance was emptied (`oldbalanceOrg - amount == 0`).
* `dest_no_update`: Flagging transactions where the recipient's balance remained identical post-transaction (`oldbalanceDest == newbalanceDest`).

While `full_drain` accounted for **97.7% of fraud events** and **0.00% of legitimate volume**, it was diagnosed as an artifact of PaySim’s underlying simulator rather than an adversarial invariant.

### 3. Model Leakage Diagnosis (Model A vs. Model B)

| Model Configuration | Recall (Fraud) | Precision (Fraud) | PR-AUC | Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** (Includes `full_drain`) | ~99.7% | ~100.0% | 0.9957 | **Severely Leaky**: Overfit to synthetic simulation rules. |
| **Model B** (Honest Baseline) | **86.6%** | **43.5%** | **0.8564** | **Deployable**: Forces learning from raw continuous deltas. |

### 4. Regulatory Explainability (SHAP)
Using `shap.TreeExplainer`, feature impact vectors were extracted for Model B:
* The model independently learned the balance-depletion pattern via `orig_balance_delta` and `oldbalanceOrg` without relying on the hard-coded `full_drain` indicator.
* Individual alert summaries provide deterministic, compliance-ready explanations for review operations.

### 5. Production Threshold Optimization
Default classification cutoffs ($p = 0.50$) generate excessive false alarms. The threshold was systematically tuned against precision constraints:

| Precision Target | Attainable Recall | Calibrated Threshold | Operational Context |
| :--- | :--- | :--- | :--- |
| **99.0%** | 65.4% | 0.904 | Minimal review capacity; highest cost of manual intervention. |
| **95.0%** | **71.0%** | **0.864** | **Selected Production Cutoff**: Optimal balance of review volume to fraud capture. |
| **90.0%** | 75.6% | 0.824 | Moderate fraud prevention prioritization. |
| **75.0%** | 80.2% | 0.713 | Aggressive risk control. |
| **50.0%** | 84.8% | 0.550 | Standard model thresholding; review queue bottleneck. |


## Executive Summary & Business Impact

| Metric / Decision Variable | Baseline Rule (`isFlaggedFraud`) | Honest Production Model (Model B @ 0.86 Threshold) | Net Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Fraud Detection Rate (Recall)** | 0.19% (16 / 8,213) | **71.0%** (154 / 217 in Test) | **~370x improvement** |
| **Operational Precision** | ~100% | **95.0%** | 1 false alert per 20 flags |
| **False Positive Rate (FPR)** | ~0.000% | **0.011%** (8 false alerts in 74,783) | Minimal manual review bloat |
| **Fraud Prevented (Test Set)** | ~₹0.23 Cr | **~₹22.6 Cr** (Avg ₹14.68L / fraud) | **+₹22.37 Cr recovered** |
| **Remaining Loss Exposure** | ~₹120.3 Cr | **~₹9.2 Cr** (63 missed fraud cases) | Drastic risk mitigation |



## Honest Limitations & Production Considerations

*Synthetic Nature:* PaySim does not reflect modern UPI features (device fingerprinting, VPA aliases, SIM binding, or tokenized flows).

*Absence of Temporal Splits:* The pipeline uses stratified random sampling. A live production engine requires out-of-time (OOT) validation splits to track adversarial drift.

*Financial Sizing:* Dollar/Rupee fraud-loss figures represent sample accounting derived from test set totals to demonstrate metric conversion, not empirical real-world losses.
