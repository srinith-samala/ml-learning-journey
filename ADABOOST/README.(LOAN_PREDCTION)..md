# AdaBoost Classifier

Loan Prediction dataset, same pipeline/`random_state=42` as the XGBoost and Gradient Boosting runs for direct comparison — the third and final Boosting-family algorithm covered, genuinely different in mechanism from the other two.

AI usage disclosure: Used Claude for concept explanation (how AdaBoost's weight-adjustment mechanism differs from Gradient Boosting's residual-fitting) and code review across several debugging rounds. Code was written and run independently.

## Pipeline followed

Identical to XGBoost/Gradient Boosting notebooks: Import → Load → Explore → Drop `Loan_ID`, `Married`, `Dependents` → Missing Values (mode for categorical, median for `LoanAmount`) → Label Encoding (separate encoder per column) → Define X, y → Train-Test Split (`random_state=42`) → No scaling (tree/stump-based) → Baseline `AdaBoostClassifier` → Tune `n_estimators` via loop (10 to 200) → Retrain at best `n_estimators` → Metrics → Feature Importance.

## Key concept: AdaBoost vs Gradient Boosting

Both are sequential Boosting algorithms, but they correct errors differently:

- **Gradient Boosting / XGBoost:** each new tree is trained to predict the *residual* (numeric error) of all previous trees combined. The target changes each round; the data stays the same.
- **AdaBoost:** the target never changes — instead, **sample weights** change. Every sample starts with equal weight (`1/N`). After each weak learner (a "stump" — a `DecisionTreeClassifier(max_depth=1)`, confirmed via `model.estimator_`) is trained, misclassified samples get their weight increased and correctly-classified samples get theirs decreased, so the next learner is forced to focus harder on the previously-hard cases.

**The maths:** each learner gets a weighted error `ε`, which produces an importance score `α = ½·ln((1−ε)/ε)` — a learner that's exactly as good as random guessing (`ε=0.5`) gets `α=0` (no vote at all), and a learner worse than random gets a *negative* vote (its prediction gets inverted). Final prediction is `sign(Σ αₜ·hₜ(x))` — a weighted vote, not a simple majority.

**Why AdaBoost needs clean data:** the exponential weight update (`exp(±α)`) means outliers/noisy samples can rapidly balloon in weight across rounds, pulling later learners toward fitting noise rather than genuine patterns — more fragile to dirty data than Gradient Boosting's more linear-ish residual approach.

## `algorithm` parameter

`model.algorithm` returned `'SAMME.R'` (sklearn's current default), which is flagged as deprecated in this sklearn version (`FutureWarning`, to be removed in 1.6, replaced by `'SAMME'`). Documenting this for future reference — notebook will need `algorithm='SAMME'` explicitly once upgraded.

## Mistakes made (and fixed) across this notebook

1. **Passed a `range()` object directly as `n_estimators`** instead of using it inside a loop — `n_estimators` expects a single integer, not a range; this raised `InvalidParameterError`.

2. **Hardcoded `n_estimators=75` as a guess** instead of reading the actual best value off the tuning loop's `test_scores` array — fixed by computing `best_n = list(n_ranges)[np.argmax(test_scores)]` to pick the value programmatically rather than eyeballing the graph.

3. **Forgot to re-run metrics on the retrained best-`n_estimators` model** — after fixing mistake #2 and retraining at `best_n=10`, initially jumped straight to `feature_importances_` without re-computing the confusion matrix/accuracy/classification report, so the "final" results were still showing the old baseline (`n_estimators=50`) model's numbers. Fixed by re-running the full metrics block against the correctly-tuned model.

## Choosing `n_estimators`

Looped from 10 to 200 in steps of 10, plotted train vs test accuracy. Best test accuracy came at the very first value tried, **`n_estimators=10`** — fewer, simpler stumps generalized better than more on this small dataset.

## Final Results (n_estimators=10)

```
Confusion Matrix: [[18 25]
                    [ 2 78]]
Accuracy: 78.0%
Train Score: 0.825 | Test Score: 0.780 (small gap — low overfitting)

              precision  recall  f1-score  support
0 (Rejected)      0.90    0.42      0.57       43
1 (Approved)      0.76    0.97      0.85       80
```

## Feature Importance (n_estimators=10)

```
LoanAmount           0.30
Property_Area        0.20
CoapplicantIncome     0.20
Loan_Amount_Term     0.10
ApplicantIncome      0.10
Credit_History       0.10
Gender, Education, Self_Employed   0.00
```

With only 10 stumps total, importance values are coarse (each stump contributes roughly a 0.1 "chunk"). Notably, `Credit_History` — the dominant feature in every other algorithm tried (0.277–0.767 range) — only scores 0.10 here, with `LoanAmount` and `Property_Area` taking the lead instead. This is a distinctly different pattern from every prior model on this dataset.

## Comparison — all three Boosting algorithms (identical dataset/split/random_state)

| Algorithm | Accuracy | Train/Test Gap | Class 0 Recall |
|---|---|---|---|
| XGBoost | 76.4% | smaller | 0.44 |
| Gradient Boosting | 74.8% | 0.892 vs 0.748 (large) | 0.42 |
| **AdaBoost** | **78.0%** | 0.825 vs 0.780 (small) | 0.42 |

AdaBoost came out as the best-performing and best-generalizing Boosting algorithm on this dataset — surprising given it's the conceptually "simplest" of the three (no gradient-based residual fitting, just weighted voting over stumps). A good reminder that more sophisticated engineering (XGBoost) doesn't guarantee better results on every dataset — AdaBoost's different error-correction mechanism happened to suit this particular small, Credit_History-light-weighted outcome better here.

## Full comparison — all nine algorithms tried on Loan Prediction

| Algorithm | Accuracy | Class 0 Recall |
|---|---|---|
| Logistic Regression | 78.9% | 0.42 |
| Decision Tree | 77.2% | 0.42 |
| Random Forest | 75.6% | 0.42 |
| KNN | 78.0% | 0.42 |
| Naive Bayes | 78.0% | 0.42 |
| XGBoost | 76.4% | 0.44 |
| Gradient Boosting | 74.8% | 0.42 |
| AdaBoost | 78.0% | 0.42 |

Eight of nine algorithms land on the identical 0.42 minority-class recall — strong, repeated confirmation that this dataset's class imbalance (~69%/31%) is the real ceiling, not any particular algorithm's weakness.