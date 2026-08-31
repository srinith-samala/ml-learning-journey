# Naive Bayes — Diabetes Prediction (Practice Dataset)

Second Naive Bayes dataset, done independently. Pima Indians Diabetes Dataset (768 rows, all-numeric features) — chosen as a "classic" GaussianNB use case: continuous medical measurements predicting a binary outcome, no encoding required.

AI usage disclosure: Used Claude for dataset guidance and a data-quality check technique (identifying disguised missing values). Code was written and run independently — no bugs on this run.

## Pipeline followed

Import → Load → Explore → Checked for disguised missing values (see below) → Define X, y (no encoding needed — all features already numeric) → Train-Test Split → Scaling (MinMaxScaler) → Create Model (`GaussianNB`) → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report

## Data-quality check: disguised missing values

Learned that `isnull().sum()` only catches explicit `NaN` — it can't catch missing values that were encoded as something else, like `0` in a column where zero isn't biologically possible (e.g. `BloodPressure = 0`, `Glucose = 0`). Checked this by counting zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`:

```python
print((df['Glucose'] == 0).sum())
print((df['BloodPressure'] == 0).sum())
# etc.
```

All came back `0` — this particular CSV version turned out to already be pre-cleaned (values like `BloodPressure = 72.0` and `Insulin = 169.5` showing decimal precision suggest a prior imputation pass was already done on the file before download). No column needed fixing, but the check itself is a habit worth keeping for any new numeric dataset — `describe()`'s `min` value is the fast way to spot an impossible zero.

## Results

```
Confusion Matrix: [[80 19]
                    [17 38]]
Accuracy: 76.6%

              precision  recall  f1-score  support
0 (No Diabetes)   0.82    0.81      0.82       99
1 (Diabetes)      0.67    0.69      0.68       55
```

## Comparison — Naive Bayes across datasets

| Metric | Loan Prediction | Diabetes |
|---|---|---|
| Accuracy | 78.0% | 76.6% |
| Minority class recall | 0.42 | **0.69** |
| Class split | ~69/31 | ~65/35 |

Much more balanced recall here than on Loan Prediction, despite a similar overall class split. This lines up with what GaussianNB is actually built for — continuous numeric features with a roughly normal distribution, which is exactly the shape of this dataset (medical measurements), versus Loan Prediction's mix of categorical and skewed numeric features. Best-balanced Naive Bayes result so far.

## Next up

Planning to try Naive Bayes on a text dataset (SMS Spam Collection) next — expecting this to show its real strength, since Naive Bayes is the standard choice for high-dimensional text classification (hundreds/thousands of word-features), unlike this dataset's 8 numeric columns.