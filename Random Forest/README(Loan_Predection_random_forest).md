# Random Forest Classifier

Loan Prediction dataset (same as Logistic Regression and Decision Tree) — used to compare an ensemble method against the single-tree and linear approaches already tried.

AI usage disclosure: Used Claude for concept explanations, roadmap structuring, and code review. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding, separate encoder per column) → Define X, y → Train-Test Split → (Scaling skipped — tree-based, not distance-based) → Create Model (`RandomForestClassifier`) → Train Model → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance

## Key concepts

**Bagging (Bootstrap Aggregating):** each tree in the forest trains on a random sample of rows drawn *with replacement* from the training data, so every tree sees a slightly different dataset (some rows repeated, some left out).

**Feature randomness:** at every split, each tree only gets to consider a random subset of columns, not all of them — this stops one dominant feature (`Credit_History`) from taking over every tree, forcing trees to also learn from other columns.

Together these force the individual trees to be genuinely different from each other, so their combined vote/average cancels out individual overfitting and noise — unlike a single Decision Tree, `max_depth` isn't critical here because averaging across many trees already controls overfitting.

**Trade-off:** loses the single-tree interpretability — `plot_tree()` doesn't meaningfully apply to 100 trees at once. `feature_importances_` still works (averaged across all trees) but you can't trace one clean decision path the way you can with a single Decision Tree.

## Mistakes I made (and fixed)

1. **Forgot to fill `Loan_Amount_Term` missing values** — filled `Gender`, `Self_Employed`, `LoanAmount`, `Credit_History` but missed this column (14 missing values). `model.fit()` didn't error out (this sklearn version tolerates some missing values in tree-based models), but leaving it unfilled is bad practice — fixed with `.fillna(mode()[0])`. Accuracy improved slightly (73.98% → 75.6%) after the fix.

## Results

```
Confusion Matrix: [[18 25]
                    [ 5 75]]
Accuracy: 75.6%

              precision  recall  f1-score  support
0 (Rejected)      0.78    0.42      0.55       43
1 (Approved)      0.75    0.94      0.83       80
```

## Feature Importance

```
Credit_History       0.277
ApplicantIncome       0.230
LoanAmount            0.200
CoapplicantIncome     0.122
Loan_Amount_Term      0.056
Property_Area         0.053
Gender                0.022
Education              0.021
Self_Employed           0.019
```

**Compare to the single Decision Tree**, where `Credit_History` alone took 0.767 importance and several features scored 0. Here importance spreads across more features (`ApplicantIncome`, `LoanAmount` are close behind) — direct evidence of feature randomness at work: no single tree in the forest could rely on `Credit_History` every time, so other columns got a real chance to matter.

## Comparison — all three algorithms (same dataset)

| Metric | Logistic Regression | Decision Tree (max_depth=4) | Random Forest |
|---|---|---|---|
| Test Accuracy | 78.9% | 77.2% | 75.6% |
| Class 0 Recall | 0.42 | 0.42 | 0.42 |
| Class 0 Precision | 0.95 | 0.86 | 0.78 |

**Notable takeaway:** Random Forest did *not* outperform the simpler models on this dataset. Likely because the dataset is small (614 rows) and dominated by one strong feature (`Credit_History`) — the kind of setting where a simpler model can match or beat a more complex ensemble. Random Forest's advantage tends to show up more on larger, noisier, or more non-linear datasets. All three models still show the same weakness: 0.42 recall on the minority class (Rejected), pointing to the dataset's class imbalance (~69% approved) as the real bottleneck, not the choice of algorithm.# Random Forest Classifier

Loan Prediction dataset (same as Logistic Regression and Decision Tree) — used to compare an ensemble method against the single-tree and linear approaches already tried.

AI usage disclosure: Used Claude for concept explanations, roadmap structuring, and code review. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding, separate encoder per column) → Define X, y → Train-Test Split → (Scaling skipped — tree-based, not distance-based) → Create Model (`RandomForestClassifier`) → Train Model → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance

## Key concepts

**Bagging (Bootstrap Aggregating):** each tree in the forest trains on a random sample of rows drawn *with replacement* from the training data, so every tree sees a slightly different dataset (some rows repeated, some left out).

**Feature randomness:** at every split, each tree only gets to consider a random subset of columns, not all of them — this stops one dominant feature (`Credit_History`) from taking over every tree, forcing trees to also learn from other columns.

Together these force the individual trees to be genuinely different from each other, so their combined vote/average cancels out individual overfitting and noise — unlike a single Decision Tree, `max_depth` isn't critical here because averaging across many trees already controls overfitting.

**Trade-off:** loses the single-tree interpretability — `plot_tree()` doesn't meaningfully apply to 100 trees at once. `feature_importances_` still works (averaged across all trees) but you can't trace one clean decision path the way you can with a single Decision Tree.

## Mistakes I made (and fixed)

1. **Forgot to fill `Loan_Amount_Term` missing values** — filled `Gender`, `Self_Employed`, `LoanAmount`, `Credit_History` but missed this column (14 missing values). `model.fit()` didn't error out (this sklearn version tolerates some missing values in tree-based models), but leaving it unfilled is bad practice — fixed with `.fillna(mode()[0])`. Accuracy improved slightly (73.98% → 75.6%) after the fix.

## Results

```
Confusion Matrix: [[18 25]
                    [ 5 75]]
Accuracy: 75.6%

              precision  recall  f1-score  support
0 (Rejected)      0.78    0.42      0.55       43
1 (Approved)      0.75    0.94      0.83       80
```

## Feature Importance

```
Credit_History       0.277
ApplicantIncome       0.230
LoanAmount            0.200
CoapplicantIncome     0.122
Loan_Amount_Term      0.056
Property_Area         0.053
Gender                0.022
Education              0.021
Self_Employed           0.019
```

**Compare to the single Decision Tree**, where `Credit_History` alone took 0.767 importance and several features scored 0. Here importance spreads across more features (`ApplicantIncome`, `LoanAmount` are close behind) — direct evidence of feature randomness at work: no single tree in the forest could rely on `Credit_History` every time, so other columns got a real chance to matter.

## Comparison — all three algorithms (same dataset)

| Metric | Logistic Regression | Decision Tree (max_depth=4) | Random Forest |
|---|---|---|---|
| Test Accuracy | 78.9% | 77.2% | 75.6% |
| Class 0 Recall | 0.42 | 0.42 | 0.42 |
| Class 0 Precision | 0.95 | 0.86 | 0.78 |

**Notable takeaway:** Random Forest did *not* outperform the simpler models on this dataset. Likely because the dataset is small (614 rows) and dominated by one strong feature (`Credit_History`) — the kind of setting where a simpler model can match or beat a more complex ensemble. Random Forest's advantage tends to show up more on larger, noisier, or more non-linear datasets. All three models still show the same weakness: 0.42 recall on the minority class (Rejected), pointing to the dataset's class imbalance (~69% approved) as the real bottleneck, not the choice of algorithm.# Random Forest Classifier

Loan Prediction dataset (same as Logistic Regression and Decision Tree) — used to compare an ensemble method against the single-tree and linear approaches already tried.

AI usage disclosure: Used Claude for concept explanations, roadmap structuring, and code review. Code was written and run independently.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding (Label Encoding, separate encoder per column) → Define X, y → Train-Test Split → (Scaling skipped — tree-based, not distance-based) → Create Model (`RandomForestClassifier`) → Train Model → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Feature Importance

## Key concepts

**Bagging (Bootstrap Aggregating):** each tree in the forest trains on a random sample of rows drawn *with replacement* from the training data, so every tree sees a slightly different dataset (some rows repeated, some left out).

**Feature randomness:** at every split, each tree only gets to consider a random subset of columns, not all of them — this stops one dominant feature (`Credit_History`) from taking over every tree, forcing trees to also learn from other columns.

Together these force the individual trees to be genuinely different from each other, so their combined vote/average cancels out individual overfitting and noise — unlike a single Decision Tree, `max_depth` isn't critical here because averaging across many trees already controls overfitting.

**Trade-off:** loses the single-tree interpretability — `plot_tree()` doesn't meaningfully apply to 100 trees at once. `feature_importances_` still works (averaged across all trees) but you can't trace one clean decision path the way you can with a single Decision Tree.

## Mistakes I made (and fixed)

1. **Forgot to fill `Loan_Amount_Term` missing values** — filled `Gender`, `Self_Employed`, `LoanAmount`, `Credit_History` but missed this column (14 missing values). `model.fit()` didn't error out (this sklearn version tolerates some missing values in tree-based models), but leaving it unfilled is bad practice — fixed with `.fillna(mode()[0])`. Accuracy improved slightly (73.98% → 75.6%) after the fix.

## Results

```
Confusion Matrix: [[18 25]
                    [ 5 75]]
Accuracy: 75.6%

              precision  recall  f1-score  support
0 (Rejected)      0.78    0.42      0.55       43
1 (Approved)      0.75    0.94      0.83       80
```

## Feature Importance

```
Credit_History       0.277
ApplicantIncome       0.230
LoanAmount            0.200
CoapplicantIncome     0.122
Loan_Amount_Term      0.056
Property_Area         0.053
Gender                0.022
Education              0.021
Self_Employed           0.019
```

**Compare to the single Decision Tree**, where `Credit_History` alone took 0.767 importance and several features scored 0. Here importance spreads across more features (`ApplicantIncome`, `LoanAmount` are close behind) — direct evidence of feature randomness at work: no single tree in the forest could rely on `Credit_History` every time, so other columns got a real chance to matter.

## Comparison — all three algorithms (same dataset)

| Metric | Logistic Regression | Decision Tree (max_depth=4) | Random Forest |
|---|---|---|---|
| Test Accuracy | 78.9% | 77.2% | 75.6% |
| Class 0 Recall | 0.42 | 0.42 | 0.42 |
| Class 0 Precision | 0.95 | 0.86 | 0.78 |

**Notable takeaway:** Random Forest did *not* outperform the simpler models on this dataset. Likely because the dataset is small (614 rows) and dominated by one strong feature (`Credit_History`) — the kind of setting where a simpler model can match or beat a more complex ensemble. Random Forest's advantage tends to show up more on larger, noisier, or more non-linear datasets. All three models still show the same weakness: 0.42 recall on the minority class (Rejected), pointing to the dataset's class imbalance (~69% approved) as the real bottleneck, not the choice of algorithm.