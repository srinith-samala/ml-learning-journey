# Naive Bayes (GaussianNB) Classifier

Loan Prediction dataset (same as all previous algorithms) — used to compare a probability-based algorithm against equation-based, tree-based, and distance-based approaches already tried.

AI usage disclosure: Used Claude for concept explanations (Bayes Theorem, the "naive" independence assumption), roadmap structuring, and code review. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding for binary columns + One-Hot Encoding for `Property_Area`) → Define X, y → Train-Test Split → Scaling (MinMaxScaler) → Create Model (`GaussianNB`) → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report

## Key concepts

**Bayes Theorem:** `P(A|B) = [P(B|A) × P(A)] / P(B)` — computes the probability of a class given the observed features, based on how likely those features are within each class and how common each class is overall. Worked through a classic example (rare-disease test) showing why a positive result on a rare condition can still mean a low actual probability of having it — the base rate (prior probability) matters as much as the test's accuracy.

**The "naive" assumption:** Naive Bayes assumes every feature is completely independent of every other feature, which is rarely true in reality (e.g. income and education are usually correlated). This simplification lets it just multiply each feature's individual probability instead of modeling complex interactions — which is why it's so fast, and why it still works well in practice even though the assumption is technically wrong (the relative ranking between classes tends to stay correct even if the exact probability values are off).

**No `random_state` needed** — unlike KNN, Decision Tree, and Random Forest, there's no randomness in Naive Bayes' calculation. It's a direct probability computation, so it always gives the same result on the same data.

## Where Naive Bayes actually shines (not fully visible on this dataset)

Naive Bayes' real strength is text classification (spam detection, sentiment analysis) — datasets with hundreds or thousands of features (words), where KNN would be too slow (distance calculations don't scale well to high dimensions) and where Naive Bayes' simple multiplication stays fast regardless of feature count. This numeric/categorical Loan dataset doesn't showcase that advantage — planning to try Naive Bayes on a text dataset (SMS Spam Detection) next to see the difference.

## Mistakes / notes

1. **Missing `drop_first=True` in `pd.get_dummies()`** — created all 3 `Property_Area` columns instead of dropping one (the dummy variable trap avoided in earlier notebooks). Didn't hurt results here since Naive Bayes treats features independently anyway (unlike Linear/Logistic Regression, where this would matter more), but keeping `drop_first=True` for consistency going forward.

2. **Filled `Credit_History` (a binary column) with `.mean()` instead of `.mode()[0]`** — same design inconsistency noted in the KNN notebook. Didn't affect results, but `.mode()[0]` is the more correct choice for categorical/binary columns.

## Results

```
Confusion Matrix: [[18 25]
                    [ 2 78]]
Accuracy: 78.0%

              precision  recall  f1-score  support
0 (Rejected)      0.90    0.42      0.57       43
1 (Approved)      0.76    0.97      0.85       80
```

## Comparison — all five algorithms (same dataset)

| Metric | Logistic Regression | Decision Tree | Random Forest | KNN | Naive Bayes |
|---|---|---|---|---|---|
| Test Accuracy | 78.9% | 77.2% | 75.6% | 78.0% | 78.0% |
| Class 0 Recall | 0.42 | 0.42 | 0.42 | 0.42 | **0.42** |

All five algorithms — spanning four completely different paradigms (linear equations, tree splits, distance, probability) — land on the exact same 0.42 recall for the minority class. This is now strong, repeated evidence that the ceiling here is the dataset's class imbalance (~69% Approved / 31% Rejected), not a weakness of any particular algorithm. Fixing this would require imbalance-specific techniques (`class_weight='balanced'`, resampling like SMOTE) rather than switching models — noted for later.