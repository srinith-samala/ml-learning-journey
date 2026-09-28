# XGBoost Classifier

Loan Prediction dataset (same as all previous algorithms) — first Boosting-family algorithm, compared against the equation-based, tree-based (Bagging), distance-based, and probability-based approaches already tried.

AI usage disclosure: Used Claude for concept explanations (Boosting vs Bagging, sequential residual correction, the Gradient Descent connection), roadmap structuring, and code review. Code was written and run independently — no bugs on this run.

## Pipeline followed

Import (`XGBClassifier` from the separate `xgboost` library, not sklearn) → Load → Explore → Handle Missing Values → Feature Selection (dropped `Loan_ID`, `Married`, `Dependents`) → Encoding (Label Encoding — XGBoost is tree-based, same reasoning as Decision Tree/Random Forest) → Define X, y → Train-Test Split → (Scaling skipped — tree-based) → Create Model (`XGBClassifier`) → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance

## Key concept: Boosting vs Bagging

Random Forest (Bagging) builds many trees **independently and in parallel**, then averages their votes. XGBoost (Boosting) builds trees **sequentially** — each new tree is trained specifically to correct the errors (residuals) of the combined trees before it, rather than seeing the raw target again.

**Connection to Gradient Descent:** at each step, the current residual (actual − predicted) is calculated, a new tree is trained to predict that residual, and the prediction is updated as `old_prediction + (learning_rate × new_tree_prediction)`. This mirrors Gradient Descent's "take a small step in the direction that reduces error" — `learning_rate` is a fixed value set upfront that scales how much each tree's correction contributes, preventing any single tree from overcorrecting.

## Results

```
Confusion Matrix: [[19 24]
                    [ 5 75]]
Accuracy: 76.4%

              precision  recall  f1-score  support
0 (Rejected)      0.79    0.44      0.57       43
1 (Approved)      0.76    0.94      0.84       80
```

## Feature Importance

```
Credit_History       0.559
LoanAmount            0.067
Loan_Amount_Term      0.065
Property_Area         0.060
CoapplicantIncome     0.057
Self_Employed         0.054
Education             0.051
ApplicantIncome       0.050
Gender                0.038
```

`Credit_History` dominates again, but at a level between Decision Tree (0.767) and Random Forest (0.277) — sits in between the two extremes, likely because Boosting's sequential correction lets other features contribute once the biggest error pattern from `Credit_History` alone has already been addressed by early trees.

## Comparison — all six algorithms (same dataset)

| Algorithm | Accuracy | Class 0 Recall |
|---|---|---|
| Logistic Regression | 78.9% | 0.42 |
| Decision Tree | 77.2% | 0.42 |
| Random Forest | 75.6% | 0.42 |
| KNN | 78.0% | 0.42 |
| Naive Bayes | 78.0% | 0.42 |
| XGBoost | 76.4% | **0.44** |

XGBoost is the first algorithm to move the minority-class recall off the identical 0.42 every other model landed on — only a small bump (0.44), but consistent with Boosting's sequential error-focus giving slightly more attention to harder-to-classify cases. Overall accuracy is still not the best on this dataset (Logistic Regression remains ahead at 78.9%) — another reminder that XGBoost's advantage tends to show up more on larger, more complex datasets than this 614-row one, which is dominated by a single strong feature.