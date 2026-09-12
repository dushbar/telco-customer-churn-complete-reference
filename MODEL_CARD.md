# Model Card — Telco Customer Churn Classifier

## Overview
- **Purpose:** Predict the probability that an active telecom customer churns, to prioritize retention outreach.
- **Version:** 1.0.0
- **Model type:** XGBoost classifier (gradient-boosted trees), tuned via `RandomizedSearchCV`.
- **Training data:** IBM/Kaggle Telco Customer Churn dataset — 7,043 customers, ~26.5% churn rate, a single historical snapshot (not a time series).

## Intended use
Ranking and flagging active customers by churn risk to prioritize a retention team's limited outreach capacity. **Not** intended as the sole basis for a customer-facing decision (e.g. automatically canceling a service or changing a price) without human review.

## Performance (held-out test set, 1,409 customers)
- ROC-AUC ≈ 0.85, PR-AUC ≈ 0.66 (see Notebook 3 for full comparison across model families)
- Default operating threshold: 0.20 (business-cost-derived, see Notebook 4) — **not** the classifier default of 0.5
- Full metric suite, cumulative gains chart, and calibration diagnostics: Notebook 4

## Known limitations
- Trained on a single historical snapshot — no validation yet against how churn drivers or the customer population shift over time. PSI-based monitoring (Section 6) is the recommended way to detect when retraining is needed.
- Two structural collinearity issues were found and corrected for the *interpretable* logistic regression model (Notebook 5, Section 2) but were not obstacles for the deployed XGBoost model, which is not sensitive to them.
- The business-cost threshold depends on assumed retention-offer cost and success-rate figures (Notebook 4) that should be revisited against the real retention program's actual numbers, not left at the illustrative values used here.

## Fairness check
A subgroup performance check (Notebook 5, Section 6) across gender and senior-citizen status found no glaring disparity in recall or precision. This was a first-pass check, not a full fairness audit.

## Reproduction
`churn_pipeline.joblib` + `churn_model_metadata.json` (this directory) — see `app.py` for a minimal serving example.
