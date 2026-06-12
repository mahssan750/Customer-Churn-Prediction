<div align="center">
  
# Customer-Churn-Prediction
Identify at-risk Customers Before They Leave — Machine Learning Project Using Real e-commerce Data
---
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-27AE60?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-27AE60?style=flat)]()

**Test AUC: 0.785 · Recall: 80.97% · 4,338 customers scored · 3 models compared**

</div>

---

## 📌 The Business Problem

Every online retailer faces the same silent threat — customers who stop buying without warning. By the time the business notices, the customer is already gone.

> **Retaining a customer costs 5–7× less than acquiring a new one.**
> The earlier you identify a churner, the cheaper it is to keep them.

This project builds a machine learning system that analyses each customer's full transaction history and answers one question:

> *"Will this customer stop buying in the next 90 days?"*

Every customer receives a churn probability (0–100%) and a **risk tier** so the retention team knows exactly who to call, who to email, and who to leave alone — ranked by urgency.

---

## 🏆 Results

### What the model achieved on real data

> Trained and evaluated on the [UCI Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) — 397,924 real UK e-commerce transactions from 4,338 customers.

#### Model Comparison — Hold-Out Test Set

| Model | CV AUC | Test AUC | Avg Precision | Precision | Recall | F1 | Train Time |
|-------|--------|----------|---------------|-----------|--------|----|------------|
| Logistic Regression | 0.7915 ± 0.0171 | 0.7703 | 0.5685 | 0.5183 | 0.7855 | 0.6245 | 0.02s |
| Random Forest | 0.7958 ± 0.0164 | 0.7836 | 0.5896 | 0.5515 | 0.7785 | 0.6456 | 0.46s |
| **XGBoost** ⭐ | **0.7833 ± 0.0111** | **0.7849** | **0.5937** | 0.5223 | **0.8097** | 0.6350 | **0.26s** |

**🏆 Best model: XGBoost** — highest Test AUC (0.7849) and best Recall (80.97%)

#### What these numbers mean for the business

| Metric | Value | Plain English |
|--------|-------|---------------|
| **Test AUC: 0.785** | 78.5% | The model correctly ranks a churning customer above a safe one 78.5% of the time — far better than guessing (50%) |
| **Recall: 80.97%** | ~4 in 5 | Out of every 5 customers who actually churn, the model catches 4 of them |
| **Precision: 52.23%** | ~1 in 2 | Of every 2 customers flagged as high-risk, roughly 1 is genuinely at risk |
| **CV Std Dev: ±0.011** | Very stable | Low variance across 5 cross-validation folds — the model performs consistently, not just on lucky splits |

---

## 💰 Business Impact

### Risk Tier Breakdown — All 4,338 Customers Scored

```
  High Risk      : 1,381  (31.8%)  ███████████████████████████████
  Medium Risk    : 1,247  (28.7%)  █████████████████████████████
  Low Risk       : 1,710  (39.4%)  ███████████████████████████████████████
```

### What each tier costs and saves

| Risk Tier | Customers | Recommended Action | Est. Cost/Customer | Est. Revenue Protected |
|-----------|-----------|-------------------|-------------------|----------------------|
| 🔴 **High Risk** | 1,381 (31.8%) | Personalised call + exclusive discount | £15–25 | £200–500/customer |
| 🟡 **Medium Risk** | 1,247 (28.7%) | Automated win-back email + voucher | £3–8 | £50–150/customer |
| 🟢 **Low Risk** | 1,710 (39.4%) | Standard newsletter — no intervention needed | £0–2 | Monitor only |

### The cost of errors — why Recall matters most

In customer retention, **missing a real churner is always worse than a false alarm**:

| Error | What happens | Estimated cost |
|-------|-------------|----------------|
| **False Negative** (missed churner) | Customer leaves silently — we lose their full remaining lifetime value | £200–2,000+ per customer |
| **False Positive** (false alarm) | We send an unnecessary offer to a safe customer | £5–25 per wasted offer |

> Since a false alarm costs ~100× less than a missed churner, the model is tuned to **maximise Recall** — it is better to over-flag than to miss real leavers.

### Return on Investment Estimate

Using conservative assumptions (avg customer value = £275, model catches 81% of churners, 30% of flagged High Risk customers are saved by intervention):

| Metric | Value |
|--------|-------|
| High Risk customers genuinely at risk | ~720 (52% of 1,381) |
| Customers saved by intervention (30%) | ~216 |
| Revenue protected (× avg £275) | **~£59,400 per cycle** |
| Total intervention cost (1,381 × £20) | ~£27,620 |
| **Net estimated value** | **~£31,780 per retention cycle** |

---

## 📊 Model Evaluation — Deep Dive

### Cross-Validation Results (Training Set Only)

The model was evaluated using 5-fold stratified cross-validation on the training set — meaning the test set was never touched during training. This gives a reliable, unbiased estimate of real-world performance.

```
Class distribution — active: 2,314 | churned: 1,156
XGBoost scale_pos_weight: 2.00  (corrects for 2:1 class imbalance)

[Logistic Regression]  CV AUC: 0.7915 ± 0.0171   fit in 0.02s
[Random Forest]        CV AUC: 0.7958 ± 0.0164   fit in 0.46s
[XGBoost]              CV AUC: 0.7833 ± 0.0111   fit in 0.26s
```

**Why low standard deviation matters:** The ±0.011 on XGBoost means across all 5 folds, performance barely varies. A model with high std dev (e.g. ±0.05) would be unstable — performing well on some data splits and poorly on others. This model is consistent.

### Why XGBoost Wins

XGBoost achieves the highest Test AUC **and** highest Recall simultaneously:

- **Highest Recall (80.97%):** Catches more real churners than any other model — the most important metric in this business context
- **Highest Test AUC (0.7849):** Best overall ability to rank churners above non-churners
- **Highest Avg Precision (0.5937):** Best area under the Precision-Recall curve
- **Fast training (0.26s):** Efficient enough to retrain monthly on updated data
- **Stable CV (±0.011):** Most consistent across cross-validation folds

### Class Imbalance Handling

The dataset has a 2:1 ratio (2 active customers per 1 churned). Without correction, a naive model could achieve 67% accuracy by predicting "active" for everyone — which would be useless.

Three techniques were applied:
- `class_weight="balanced"` for Logistic Regression and Random Forest
- `scale_pos_weight=2.00` for XGBoost (computed from actual training distribution)
- ROC-AUC as the primary metric (insensitive to class imbalance)

---

## 🧠 The Machine Learning

### Dataset

| Attribute | Value |
|-----------|-------|
| Source | UCI Online Retail Dataset — real UK e-commerce transactions |
| Period | December 2010 – December 2011 |
| Raw transaction rows | ~397,000 |
| Unique customers | 4,338 |
| **Churn rate** | **33.3%** (1 in 3 customers churned) |
| Class ratio | 2:1 (active vs churned) |
| Train / test split | 80% / 20% (stratified) |

### How Churn Is Defined

```
Churned = 1   if   days_since_last_purchase > 90
Churned = 0   otherwise
```

This is a **business decision**. The 90-day threshold is configurable — changing it redefines the entire problem without touching any code.

### The 10 Features the Model Learns From

Each feature summarises a different dimension of customer behaviour — computed by aggregating all of a customer's transactions into a single row.

| # | Feature | What it measures |
|---|---------|-----------------|
| 1 | `TotalRevenue` | Total money spent across all purchases (lifetime value proxy) |
| 2 | `TotalOrders` | Number of distinct shopping trips (loyalty frequency) |
| 3 | `TotalItems` | Total units purchased (purchase volume) |
| 4 | `AvgLineItemRevenue` | Average revenue per product line (price sensitivity) |
| 5 | `BasketValue` | Average spend per trip = TotalRevenue / TotalOrders |
| 6 | `RevenuePerItem` | Average unit price = TotalRevenue / TotalItems |
| 7 | `UniqueProducts` | Breadth of product catalogue engagement |
| 8 | `UniqueCountries` | Number of shipping destinations |
| 9 | `Tenure` | Days from first to last purchase (customer lifespan) |
| 10 | `AvgDaysBetween` | Typical gap between purchases (purchase frequency proxy) |

### ⚠️ Critical: Why `Recency` Is Excluded (Data Leakage Prevention)

The churn label is **defined as** `Recency > 90 days`. Including `Recency` as a feature would cause **direct data leakage** — the model would simply learn:

```
if Recency > 90 → predict Churn
```

...instead of learning real behavioural patterns. In production, this would fail completely because future customers haven't reached 90 days yet.

`Recency`'s predictive signal is captured **indirectly** through:
- `AvgDaysBetween` — the customer's habitual purchase frequency
- `Tenure` — how long they have been engaged overall

---
