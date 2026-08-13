# Decision Tree Classifier

Loan Prediction dataset (same as Logistic Regression) — used to compare how a tree-based model handles the same data differently from a linear model.

AI usage disclosure: Used Claude for roadmap structuring, concept explanations, and debugging help. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding only) → Define X, y → Train-Test Split → (Scaling skipped — not needed for tree-based models) → Create Model → Train Model → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance → Tree Visualization

## Key differences from Logistic Regression

- **Encoding:** Label Encoding used for all categorical columns instead of One-Hot — trees split on comparisons (`<=`), not weighted sums, so false ordering isn't a real risk here the way it is for linear models.
- **Scaling:** Not needed at all — trees aren't distance-based, so MinMaxScaler was dropped from the pipeline entirely.
- **Interpretability:** `feature_importances_` and `plot_tree()` give direct visibility into what the model actually learned — not available with Logistic Regression's coefficients in the same intuitive way.

## Mistakes I made (and fixed)

1. **Wrong import path** — tried `from sklearn.linear_model import DecisionTreeClassifier`. Wrong module; tree-based models live in `sklearn.tree`.

2. **Duplicate/leftover cell corrupted a column** — an old, unfixed cell (`df['LoanAmount'].fillna(df['LoanAmount'].median)` — missing `()`) was still present and executing before the corrected version below it. The corrected cell ran on already-corrupted data, so the fix appeared to fail. Lesson: search the whole notebook for duplicate versions of a line before assuming a fix didn't work.

3. **`train_test_split()` output order swapped** — wrote `X_train, y_train, X_test, y_test = train_test_split(...)`, but the function always returns `X_train, X_test, y_train, y_test` in that fixed order. This silently put test features into `y_train` and training labels into `X_test`, which showed up as a `ValueError: Input y contains NaN` (because `y_train` was actually holding a features dataframe, not the target).

4. **Started with no `max_depth` restriction** — first run scored 1.0 on training data but only 0.707 on test data, a clear overfitting signal (confirmed by comparing `model.score(X_train, y_train)` vs test accuracy). Setting `max_depth=4` brought training accuracy down to 0.83 and, notably, *improved* test accuracy to 0.772 — restricting depth didn't just reduce overfitting, it produced a better-generalizing model.

## Results (max_depth=4)

```
Confusion Matrix: [[18 25]
                    [ 3 77]]
Accuracy: 77.2%
Train accuracy: 83.3% (vs 100% unrestricted — overfitting fixed)

              precision  recall  f1-score  support
0 (Rejected)      0.86    0.42      0.56       43
1 (Approved)      0.75    0.96      0.85       80
```

## Feature Importance

```
Credit_History       0.767
ApplicantIncome       0.088
CoapplicantIncome     0.054
Loan_Amount_Term      0.041
Property_Area         0.036
Gender                0.013
Self_Employed          0.000
Education               0.000
LoanAmount              0.000
```

`Credit_History` dominates at ~77% importance — consistent with what Logistic Regression's behavior already suggested when this column was dropped earlier (model collapsed to predicting majority class). The tree confirms it mathematically, and also completely ignored `Education`, `Self_Employed`, and `LoanAmount` — it never needed them for a useful split.

## Comparison vs Logistic Regression (same dataset)

| Metric | Logistic Regression | Decision Tree (max_depth=4) |
|---|---|---|
| Test Accuracy | 78.9% | 77.2% |
| Class 0 Recall | 0.42 | 0.42 |
| Class 1 Recall | 0.99 | 0.96 |

Both models struggle similarly with the minority class (Rejected) — likely a natural class imbalance issue (~69% approved in the dataset) rather than a model-specific weakness. Worth revisiting with `class_weight='balanced'` or comparing against Random Forest next.