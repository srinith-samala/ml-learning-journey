# Random Forest — Telco Customer Churn (Practice Dataset)

Second Random Forest dataset, done independently. Telco Customer Churn dataset (Kaggle/blastchar, 7043 rows, 21 columns) — chosen for a much larger row count and many more categorical columns than the Loan Prediction dataset, for heavier encoding practice.

AI usage disclosure: Used Claude for dataset guidance and a debugging tip on the `TotalCharges` column type issue. Code was written and run independently — no bugs on this run.

## Pipeline followed

Import → Load → Explore → Drop `customerID` (identifier, no predictive value) → Fix `TotalCharges` (see below) → Handle Missing Values → Encoding (Label Encoding, separate encoder per column — 15 categorical columns) → Define X, y → Train-Test Split → (Scaling skipped) → Create Model (`RandomForestClassifier`) → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance

## Key issue handled: `TotalCharges` dtype

`TotalCharges` looked numeric but was stored as `object` (text) — some rows had an empty string instead of a number. Fixed with:
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```
`errors='coerce'` converts anything that can't become a number into `NaN` instead of raising an error. This revealed 11 missing values, which were then filled with the median like any other numeric column.

## Results

```
Confusion Matrix: [[945  91]
                    [197 176]]
Accuracy: 79.6%

              precision  recall  f1-score  support
0 (No Churn)      0.83    0.91      0.87     1036
1 (Churn)         0.66    0.47      0.55      373
```

Better accuracy than Random Forest on Loan Prediction (75.6%) — likely helped by the much larger dataset (7043 vs 614 rows). Class 1 (Churn) recall is still weak (0.47) — the dataset is imbalanced (~73% No Churn / 27% Churn), same pattern seen on the Loan dataset's minority class.

## Feature Importance

```
TotalCharges       0.190
MonthlyCharges     0.178
tenure             0.157
Contract           0.077
PaymentMethod      0.050
OnlineSecurity     0.047
TechSupport        0.044
(remaining 12 features, all > 0, spread across 0.005–0.028)
```

**Key difference from Loan Prediction:** no single dominant feature here. On the Loan dataset, `Credit_History` alone carried 27–77% importance depending on the model. Here, the top 3 features combined only reach ~52%, and every single feature contributes something (nothing at 0.000). This suggests churn is driven by a genuine combination of factors — billing amount, how long someone's been a customer, and contract type — rather than one dominant signal, which is a more realistic pattern for a lot of real-world business problems than the Loan dataset's setup.

**Business read:** customers with short tenure, month-to-month contracts, and higher monthly charges appear to be the highest churn risk — useful for targeting retention offers.

## Comparison — Random Forest across datasets

| Metric | Loan Prediction (614 rows) | Telco Churn (7043 rows) |
|---|---|---|
| Accuracy | 75.6% | 79.6% |
| Minority class recall | 0.42 | 0.47 |
| Dominant single feature | Credit_History (27.7%) | None — importance spread across many features |

Larger dataset + more genuinely predictive features gave Random Forest more room to perform well, consistent with the idea that ensembles tend to show their advantage more clearly on bigger, richer datasets.