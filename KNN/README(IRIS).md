# KNN — Iris Species Classification (Practice Dataset)

Second KNN dataset, done independently. Classic Iris dataset (150 rows) — chosen specifically for its first **multiclass** problem (3 species) after four binary classification datasets, and because all features are already numeric (no encoding needed for features, only for the target).

AI usage disclosure: Used Claude for dataset guidance and a code review catching one missed column drop. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Missing Values (none) → Encode target (`Species` via Label Encoding, 3 classes → 0/1/2) → Drop `Id` (identifier, no predictive value) and `Species` from features → Define X, y → Train-Test Split → Scaling (MinMaxScaler) → Create Model (`KNeighborsClassifier`) → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → K-tuning loop

## Mistake I made (and fixed)

Forgot to drop the `Id` column from the feature set — only dropped `Species` (the target), so `Id` was being fed into the model as a feature. Didn't affect the result here since the Iris dataset is cleanly separable enough that an irrelevant column didn't hurt, but it's still bad practice to leave identifier columns in `X` — fixed by dropping both `Species` and `Id`.

## Results

```
Confusion Matrix: [[10  0  0]
                    [ 0  9  0]
                    [ 0  0 11]]
Accuracy: 100%

              precision  recall  f1-score  support
0 (Setosa)        1.00    1.00      1.00       10
1 (Versicolor)    1.00    1.00      1.00        9
2 (Virginica)     1.00    1.00      1.00       11
```

Every K from 1 to 20 gave 100% test accuracy — no variation at all.

## Important lesson: why 100% accuracy here is NOT a red flag

Normally, 100% accuracy on a real dataset is a warning sign — usually a symptom of data leakage (e.g. the target accidentally left in the features) or severe overfitting. Here it's genuinely legitimate:

- The Iris dataset is famous specifically because its 3 species are extremely well-separated by these 4 measurements — it's one of the "easiest" classification problems in ML, used for teaching rather than representing real-world difficulty.
- Small dataset (150 rows, only 30 in the test split) makes a perfect score easier to hit by chance too.
- Getting 100% consistently across every K value (not just one lucky K) confirms this isn't a fluke — the classes really are that cleanly separable.

**Takeaway for future datasets:** the Loan Prediction and Telco Churn datasets gave realistic 70–85% accuracy — that's the normal range for real-world data. If a real dataset ever shows 100% accuracy, the right instinct is to be suspicious first (check for leaked target columns, train/test overlap) rather than assume the model is simply excellent. Iris is the rare legitimate exception.

## First multiclass confusion matrix

This is the first dataset with more than 2 classes, so the confusion matrix is 3×3 instead of 2×2. Reading it is the same idea as binary — the diagonal (10, 9, 11) holds every correctly classified sample, and every off-diagonal cell (which would represent one species being confused for another) is 0.

## Comparison — KNN across datasets

| Metric | Loan Prediction (binary, 614 rows) | Iris (multiclass, 150 rows) |
|---|---|---|
| Best K | 9 | Any K (1–20 all tied at 100%) |
| Accuracy | 78.0% | 100% |
| Classes | 2 (imbalanced ~69/31) | 3 (balanced, ~50 each) |

Iris being balanced across 3 classes (unlike Loan Prediction's skew) plus its clean feature separation explains the dramatic accuracy gap — a reminder that dataset difficulty matters far more than which algorithm is used.