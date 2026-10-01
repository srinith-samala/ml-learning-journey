# AdaBoost — Telco Customer Churn (Practice Dataset)

Second AdaBoost dataset, done independently. Telco Customer Churn (Kaggle/blastchar, 7043 rows) — same dataset used for Random Forest, chosen specifically to compare Boosting vs Bagging on a larger, richer dataset than Loan Prediction.

AI usage disclosure: Used Claude for a debugging fix (a copy-paste error) and code review. Core pipeline, tuning loop, and encoding were done independently.

## Pipeline followed

Same as the Loan Prediction AdaBoost notebook: Import → Load → Drop `customerID` → Fix `TotalCharges` dtype (`pd.to_numeric(errors='coerce')`, since some rows had empty strings) → Label Encoding (15 categorical columns, separate encoder each) → Define X, y → Train-Test Split → Baseline `AdaBoostClassifier` → Tune `n_estimators` via loop (10–200) → Retrain at best value → Metrics → Feature Importance.

## Bug fixed

**Copy-paste error filling the wrong column:**
```python
df['TotalCharges'] = df['TechSupport'].fillna(df['TotalCharges'].median())  # WRONG
```
This overwrote `TotalCharges` with `TechSupport`'s text values (`'Yes'`/`'No'`) instead of filling `TotalCharges`'s own missing values. `isnull().sum()` still showed 0 missing values afterward (since `TechSupport` itself had no nulls), which hid the mistake until `model.fit()` threw `ValueError: could not convert string to float: 'No'`. Fixed by correcting the fillna target:
```python
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

## Choosing n_estimators

Looped 10 to 200 in steps of 10, picked the best via `np.argmax(test_scores)` rather than guessing — landed on **`n_estimators=130`**, a meaningfully larger value than the Loan Prediction dataset's best (10), consistent with the idea that a bigger, richer dataset can support more boosting rounds before overfitting kicks in.

## Results (n_estimators=130)

```
Confusion Matrix: [[936 100]
                    [160 213]]
Accuracy: 81.5%
Train Score: 0.811 | Test Score: 0.815 (test slightly above train — negligible overfitting)

              precision  recall  f1-score  support
0 (No Churn)      0.85    0.90      0.88     1036
1 (Churn)         0.68    0.57      0.62      373
```

This is the best result on the Telco Churn dataset across both ensemble methods tried (Random Forest and AdaBoost).

## Feature Importance

```
TotalCharges      0.377
MonthlyCharges    0.223
tenure            0.200
Contract          0.062
PaymentMethod     0.031
OnlineSecurity    0.023
(remaining features, small or 0)
```

## Comparison — AdaBoost vs Random Forest (same dataset)

| Metric | Random Forest | AdaBoost |
|---|---|---|
| Accuracy | 79.6% | **81.5%** |
| Churn (minority) Recall | 0.47 | **0.57** |
| Top-3 feature importance combined | ~52.5% | **~80%** |

AdaBoost outperformed Random Forest on both accuracy and minority-class recall here — a reversal from the Loan Prediction dataset, where AdaBoost and Random Forest were closer and Random Forest actually trailed several simpler models. On this larger (7043-row), richer dataset, Boosting's sequential error-focus clearly paid off.

**Feature importance also tells a different story than Bagging:** AdaBoost concentrated importance heavily into the top 3 features (`TotalCharges`, `MonthlyCharges`, `tenure` — ~80% combined), while Random Forest's feature randomness spread importance more evenly across many features (same top 3 only totaled ~52.5%). This matches the underlying mechanism — AdaBoost's stumps keep re-focusing on whatever most reduces weighted error, which tends to repeatedly favor the strongest few predictors, whereas Random Forest deliberately restricts feature access per split to force diversity.

## AdaBoost across both datasets

| Dataset | Rows | Best n_estimators | Accuracy | Minority Recall |
|---|---|---|---|---|
| Loan Prediction | 614 | 10 | 78.0% | 0.42 |
| Telco Churn | 7043 | 130 | 81.5% | 0.57 |

Larger dataset supported a much higher optimal `n_estimators` and produced both better accuracy and better minority-class detection — consistent with Boosting-family algorithms generally needing more data to show their full advantage.