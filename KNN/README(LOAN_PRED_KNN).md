# K-Nearest Neighbors (KNN) Classifier

Loan Prediction dataset (same as Logistic Regression, Decision Tree, Random Forest) — used to compare a distance-based algorithm against the equation-based and tree-based approaches already tried.

AI usage disclosure: Used Claude for concept explanations, roadmap structuring, and code review. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding for binary columns + **One-Hot Encoding** for `Property_Area`) → Define X, y → Train-Test Split → **Scaling (MinMaxScaler — required, unlike the tree-based models)** → Find best `K` via a loop over `n_neighbors` values → Create Model with best K → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report

## Key concept

KNN doesn't "train" an equation or build splits — it stores the training data and, for every new prediction, calculates the **Euclidean distance** to every training point, takes the `K` closest ones, and returns the majority class among them.

Because it's purely distance-based:
- **Scaling is mandatory** — without it, columns with naturally larger numbers (like `ApplicantIncome`) would dominate the distance calculation over columns with smaller ranges (like `Gender` as 0/1), regardless of actual importance.
- **One-Hot Encoding is back** for nominal columns (`Property_Area`) — same reasoning as Logistic Regression, since a raw label-encoded number would create a false sense of numeric distance between categories that don't have a real order.

## Choosing K

Looped `n_neighbors` from 1 to 20 and printed test accuracy for each, then set the final model to the K that scored highest, rather than guessing a value upfront.

```
K=1 -> 0.675   K=9  -> 0.780 (best)
K=2 -> 0.610   K=10 -> 0.772
K=3 -> 0.756   K=14 -> 0.772
...
```

`K=9` gave the best test accuracy (0.780). Small `K` values (1–2) performed noticeably worse — too sensitive to individual nearby points (closer to overfitting behavior), while K=9 balanced enough neighbors to smooth out noise without losing local signal.

## Design note (not a bug, but worth flagging for consistency)

Filled `Credit_History` (a binary 0/1 column) using `.mean()` instead of `.mode()[0]`. This produces a fractional value (e.g. 0.84) rather than a clean 0 or 1. It didn't hurt the model here since KNN just treats it as a scaled number either way, but `.mode()[0]` is the more conceptually correct choice for a categorical/binary column — used `.mean()`/`.median()` for genuinely continuous columns like `LoanAmount` instead going forward.

## Results (K=9)

```
Confusion Matrix: [[18 25]
                    [ 2 78]]
Accuracy: 78.0%

              precision  recall  f1-score  support
0 (Rejected)      0.90    0.42      0.57       43
1 (Approved)      0.76    0.97      0.85       80
```

## Comparison — all four algorithms (same dataset)

| Metric | Logistic Regression | Decision Tree (depth=4) | Random Forest | KNN (K=9) |
|---|---|---|---|---|
| Test Accuracy | 78.9% | 77.2% | 75.6% | 78.0% |
| Class 0 Recall | 0.42 | 0.42 | 0.42 | **0.42** |

**Key observation:** every algorithm gives the exact same 0.42 recall on the minority class (Rejected). This is strong evidence the bottleneck isn't the choice of algorithm — it's the dataset's class imbalance (~69% Approved / 31% Rejected). No amount of switching models fixes this on its own; it would need imbalance-specific handling (`class_weight='balanced'`, resampling techniques like SMOTE, etc.) — noted for later, not yet covered.

KNN slightly outperformed both Decision Tree and Random Forest on overall accuracy here, coming just behind Logistic Regression.