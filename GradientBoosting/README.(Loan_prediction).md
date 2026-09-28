# Gradient Boosting Classifier

Loan Prediction dataset (same as all previous algorithms) — the "unoptimized ancestor" of XGBoost, run on identical data/split to directly compare the two.

AI usage disclosure: Used Claude for concept explanation (how this differs from XGBoost) and code review. Code was written and run independently — no bugs, ran cleanly.

## Pipeline followed

Identical to the XGBoost notebook in every step except the model class — same missing value handling, same Label Encoding, same train-test split, no scaling (tree-based).

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```

## Key concept: this IS Gradient Boosting — XGBoost is its optimized version

Same core algorithm as XGBoost (sequential trees, each predicting the residual of the trees before it, combined via `learning_rate`). `GradientBoostingClassifier` is sklearn's plain, built-in implementation — it's missing the optimizations XGBoost adds on top: stronger built-in regularization, automatic missing-value handling, and internal parallelization for speed. Same underlying maths, less engineering around it.

## Results

```
Confusion Matrix: [[18 25]
                    [ 6 74]]
Accuracy: 74.8%

              precision  recall  f1-score  support
0 (Rejected)      0.75    0.42      0.54       43
1 (Approved)      0.75    0.93      0.83       80

Train Score: 0.892
Test Score: 0.748
```

## Direct comparison vs XGBoost (identical dataset, split, and hyperparameters)

| Metric | XGBoost | Gradient Boosting |
|---|---|---|
| Accuracy | 76.4% | 74.8% |
| Class 0 Recall | 0.44 | 0.42 |
| Train/Test Gap | smaller | **0.892 vs 0.748 — noticeably larger** |

This is a clean, practical confirmation of the concept: with identical `n_estimators`, `learning_rate`, and `max_depth`, XGBoost both scored higher on test accuracy *and* overfit less. The gap between train and test score is meaningfully wider here than in the XGBoost run, consistent with XGBoost's built-in regularization doing real work to control overfitting that plain Gradient Boosting doesn't have.

## Note for next time

Ran the whole pipeline in a single cell — works, but makes debugging harder if something breaks (have to re-run and scan the whole block to isolate the failing line). Splitting into separate cells per step, as in earlier notebooks, is better practice going forward.

## Comparison — all seven algorithms (same dataset)

| Algorithm | Accuracy | Class 0 Recall |
|---|---|---|
| Logistic Regression | 78.9% | 0.42 |
| Decision Tree | 77.2% | 0.42 |
| Random Forest | 75.6% | 0.42 |
| KNN | 78.0% | 0.42 |
| Naive Bayes | 78.0% | 0.42 |
| XGBoost | 76.4% | 0.44 |
| Gradient Boosting | 74.8% | 0.42 |

Gradient Boosting is the lowest-scoring model tried on this dataset so far — on a small (614-row), single-feature-dominated dataset like this, the extra model complexity doesn't pay off, and without XGBoost's regularization it overfits more than any tree-based model tried previously.