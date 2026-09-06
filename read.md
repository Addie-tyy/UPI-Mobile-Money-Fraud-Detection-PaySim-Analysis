# UPI/Mobile Money Fraud Detection — PaySim Analysis

An end-to-end fraud analytics project simulating the kind of risk-analyst work done
at a UPI-scale payments company (PhonePe, Paytm, Google Pay): SQL-driven exploration,
leakage-aware feature engineering, explainable ML, and a precision/recall tradeoff
framed as an actual business decision — not just a model accuracy number.

## Why this project

Built to demonstrate the intersection consumer-fintech companies screen analysts for:
SQL at scale (window functions, CTEs), fraud-domain reasoning, precision/recall as a
business tradeoff, model explainability for a compliance audience, and — critically —
the judgment to catch and report a model's own leakage rather than publish an
inflated accuracy number.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — the standard academic
mobile-money fraud dataset (Lopez-Rojas et al.), sourced from Kaggle. 6.3M+ real
transactions across 5 transaction types, with a ground-truth `isFraud` label and a
platform-native naive fraud flag (`isFlaggedFraud`) used here as a real baseline.

**Note on recency:** PaySim was published in 2016. Real UPI's overall success rate is
now ~99.2%, with technical decline down to ~0.7–0.8% (NPCI, 2026) — far better than
2016-era infrastructure, and UPI fraud cases in FY2025-26 ran to ~10.64 lakh cases
worth ~₹805 crore (NPCI). This project treats PaySim as a stand-in for proprietary
transaction data — the analytical *approach* generalizes even though this 2016
dataset's specific rates don't reflect 2026 UPI infrastructure.

## Method

**1. Exploratory SQL (SQLite, 6.3M rows):** fraud occurs exclusively in TRANSFER and
CASH_OUT transactions; a CTE + `RANK()` query surfaces top fraud-linked senders by
volume; a hypothesis that fraud accounts show single-transaction behavior was tested
and **rejected** — both fraud and legitimate senders average ~1.0 transactions/account
in this dataset, a structural property of PaySim's account IDs, not a fraud signal.

**2. Feature discovery:** explored PaySim's balance columns and found `full_drain`
(sender balance fully emptied — 97.7% of fraud vs. 0.0% of legitimate transactions)
and `dest_no_update` (destination balance never updates — 49.6% vs. 0.06%).

**3. Two models, deliberately compared to diagnose leakage:**

| Metric (fraud class) | Model A — with `full_drain` | Model B — without `full_drain` |
|---|---|---|
| Recall | ~99.7% | 86.6% |
| Precision | ~100% | 43.5% |
| PR-AUC | 0.9957 | 0.8564 |

Model A's near-perfect score is flagged as a red flag, not a win — `full_drain`
reflects how PaySim's simulation *generates* fraud, not how real adversarial fraud
behaves. Model B is the honest, deployable result.

**4. Explainability (SHAP):** confirms Model B independently rediscovers the
balance-drain pattern from raw features — `orig_balance_delta` and `oldbalanceOrg`
are the strongest predictors, matching the SQL findings, without ever being told the
`full_drain` rule directly.

**5. Threshold selection**, computed rather than defaulted:

| Minimum precision | Best achievable recall | Threshold |
|---|---|---|
| 99% | 65.4% | 0.904 |
| 95% | 71.0% | 0.864 |
| 90% | 75.6% | 0.824 |
| 75% | 80.2% | 0.713 |
| 50% | 84.8% | 0.550 |

**Recommended operating point: threshold 0.86 (95% precision, 71.0% recall)** — only
1 in 20 flagged transactions is a false alarm, a workload a manual review team can
act on directly.

## Results in business terms

**Baseline comparison:** PaySim's own naive rule caught only 16 of 8,213 fraud cases
(0.19%). Model B at the recommended threshold catches 71.0% — a **~370x improvement**
in detection capability.

**Cost-benefit** (test set, 217 fraud cases, avg. fraud amount ₹14.68 lakh):

| | Count | Value |
|---|---|---|
| Fraud caught | 154 | ~₹22.6 Cr prevented |
| Fraud missed | 63 | ~₹9.2 Cr still lost |
| False positives | 8 of 74,783 | 0.011% false-positive rate |

## Key findings (resume-ready)

- Explored 6.3M+ transactions via SQL (CTEs, window functions) to isolate fraud to
  2 of 5 transaction types before modeling.
- Discovered and engineered a balance-consistency feature (`full_drain`) present in
  97.7% of fraud cases vs. 0% of legitimate transactions.
- Diagnosed feature leakage by training and comparing two models, showing recall
  drops from ~99.7% to a realistic 86.6% once the leakage feature is removed —
  reported the inflated version as a finding about the dataset, not a result.
- Used SHAP to confirm the model independently rediscovers the balance-drain pattern
  from raw features, giving a compliance-ready explanation for every flag.
- Selected a production threshold (0.86) balancing 95% precision against 71% recall:
  ~₹22.6 Cr in fraud prevented, 0.011% false-positive rate, ~370x improvement over
  the platform's existing naive detection rule.

## Challenges & debugging

1. **A pandas datetime-unit bug** on an earlier synthetic version of this project —
   `.astype('int64')` on a `datetime64` column returned microseconds, not the
   nanoseconds I assumed, silently breaking a velocity-fraud rule (89% of
   transactions flagged instead of ~0.03%). Found by manually checking one account's
   raw timestamps against computed values rather than trusting the aggregate output.
2. **A backwards threshold search** when reading the precision-recall curve —
   `precision_recall_curve` orders points by increasing threshold, so searching from
   index 0 matched immediately every time and returned recall=1.0 for every
   precision target. Fixed by searching for the best recall among points meeting
   the precision bar, not by array position.
3. **A SHAP API shape change** — `TreeExplainer.shap_values()` returned a 3D array
   `(samples, features, classes)` in the installed SHAP version rather than the
   `[class_0, class_1]` list format from older documentation/tutorials. Diagnosed by
   inspecting the shape directly and indexing `shap_values[:, :, 1]` instead.

## Honest limitations

- PaySim is synthetic and dated (2016); its infrastructure/fraud rates don't reflect
  2026 UPI. Framed here as a stand-in dataset, not real UPI data.
- `full_drain`'s near-perfect separation in Model A likely reflects PaySim's
  simulation methodology rather than a pattern generalizing to adversarial,
  evolving real-world fraud.
- No temporal validation — train/test split is random, not time-based. A production
  system would validate on a strictly later time window to catch concept drift.
- Cost-benefit figures are computed directly from the test set's actual fraud value
  and are illustrative of the method, not a claim about real-world UPI fraud losses.

## Reproducing

Google Colab, PaySim via the Kaggle API (`kaggle.json` or Colab Secrets), `pandas` /
`sqlite3` / `scikit-learn` / `shap`. Store your Kaggle token in Colab's Secrets
manager as `KAGGLE_API_TOKEN` — never hardcode it in a notebook cell.
