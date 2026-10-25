# SVM (Support Vector Machine) Classifier

Loan Prediction dataset (same as all previous algorithms) — the final supervised classification algorithm in the core roadmap, bringing a margin-maximization paradigm distinct from equation-based, tree-based, distance-voting, probability-based, and boosting approaches already tried.

AI usage disclosure: Used Claude for concept explanations (maximum margin, support vectors, the kernel trick) and code review across two debugging rounds, and again for concept explanations (K-Fold, `GridSearchCV`, `Pipeline`) and code review during the revisit at the end of this README. Code was written and run independently.

## Pipeline followed

Import (`SVC` from `sklearn.svm`) → Load → Explore → Drop `Loan_ID`, `Married`, `Dependents` → Missing Values → Encoding (Label Encoding for binary columns + One-Hot for `Property_Area` — SVM calculates on raw numeric values like Logistic Regression/KNN, so One-Hot is needed again) → Define X, y → Train-Test Split → **Scaling (MinMax, mandatory)** → Baseline `SVC(kernel='rbf', C=1.0, probability=True)` → Tune `C` via loop → Retrain at best `C` → Metrics → Overfitting check.

## Key concepts

**Maximum margin:** instead of just finding *a* line that separates two classes, SVM finds the one with the widest possible gap ("margin") between itself and the closest points of each class — the idea being that a wider margin generalizes better to new data.

**Support vectors:** only the points closest to the boundary actually define it — points far from the boundary have zero influence on where the line sits. This is where the algorithm's name comes from.

**Kernel trick:** real data often isn't linearly separable (e.g. one class forming a ring around another). Rather than literally transforming data into a higher dimension (expensive), a kernel function (here, `rbf`) mathematically achieves the same effect — making the data separable in an implicit higher-dimensional space — without the explicit, costly transformation.

**`C` hyperparameter:** controls how strictly the margin is enforced. Small `C` allows more misclassified points for a wider, simpler margin (risk of underfitting); large `C` punishes misclassification harder, producing a narrower, more complex boundary (risk of overfitting). Tuned the same way as `max_depth`/`n_estimators` in earlier notebooks: loop over candidate values, pick whichever gives the best test accuracy (the method revisited below).

**No `feature_importances_`:** unlike every tree-based model tried so far, SVM doesn't expose feature importances directly (only a rough `coef_`-based workaround exists, and only for a linear kernel) — a genuinely new limitation compared to Decision Tree/Random Forest/XGBoost/AdaBoost.

## Mistakes made (and fixed)

1. **Trained the first model on unscaled data** — built `Scaled_X_train`/`Scaled_X_test` correctly but then called `model.fit(X_train, y_train)` on the raw (unscaled) data by mistake. Since SVM is margin/distance-based, this produced a completely collapsed model that only ever predicted the majority class (`[[0,43],[0,80]]`, 65% accuracy, 0.00 precision/recall on class 0) — the same failure mode seen whenever scaling is skipped for a distance-based algorithm. Fixed by fitting on `Scaled_X_train` instead.

2. **Repeated the same mistake in the final Train/Test score check** — after correctly predicting on `Scaled_X_test`, the last two lines still called `model.score(X_train, y_train)` / `model.score(X_test, y_test)` on unscaled data (leftover from before the fix), silently producing a mismatched, lower, and meaningless score (0.696/0.650) even though the confusion matrix and accuracy above it were already correct. Fixed by scoring against `Scaled_X_train`/`Scaled_X_test` consistently.

## Choosing C

Looped `C` over `[0.1, 1, 10, 100]` (log-scaled values, plotted with `plt.xscale('log')` since the values aren't evenly spaced) and picked the best via `np.argmax(test_scores)`. Best value: **`C=0.1`** — very close to the default (1.0), so results barely changed from the baseline run. Note that this picked `C` using test accuracy, which is exactly what the revisit below corrects.

## Results (C=0.1)

```
Confusion Matrix: [[18 25]
                    [ 1 79]]
Accuracy: 78.9%
Train Score: 0.815 | Test Score: 0.789 (small gap — low overfitting)

              precision  recall  f1-score  support
0 (Rejected)      0.95    0.42      0.58       43
1 (Approved)      0.76    0.99      0.86       80
```

## Comparison — all nine algorithms (Loan Prediction)

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
| **SVM** | **78.9%** | **0.42** |

SVM tied Logistic Regression for the best accuracy on this dataset (78.9%), and landed on the same 0.42 minority-class recall as eight of the nine algorithms tried (XGBoost's 0.44 is the only exception). The first explanation for this was the dataset's class imbalance (~69% / 31%); the revisit below found a likelier contributing cause, so read it before drawing a conclusion from this table. This closes out the core supervised classification roadmap (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost, Gradient Boosting, AdaBoost, SVM).

---

## Revisited: Pipeline, Cross-Validation & GridSearchCV (`svm_model_tuning_pipeline.ipynb`)

Two problems with the tuning above. First, `C` was chosen by looping and comparing **test** accuracy, so the test set influenced the choice and the reported score is best-of-N rather than an honest estimate. Second, scaling was fitted on all of `X_train` before any validation split, which leaks information. This revisit fixes both: tuning happens inside the training data using 5-fold Cross-Validation, scaling happens inside a `Pipeline`, and the test set is touched once, at the end.

### What was used

**`Pipeline` (scaler + SVC):** during Cross-Validation the scaler is fitted on each round's *training folds only* and merely applied to the validation fold. Fitting it on the whole of `X_train` first would let the validation fold's min/max influence the scaling the model sees — a leak. As a side benefit, the pipeline always takes raw `X_train`/`X_test`, so a variable like `Scaled_X_train` never exists, which removes the exact bug behind both mistakes above. Hyperparameters are addressed as `step__param` (`svm__C`, `svm__gamma`).

**`GridSearchCV` on the pipeline:** every combination in the grid is scored by 5-fold CV on `X_train`, the best is refit on all of `X_train` (`refit=True`), and `grid.predict()` uses it directly — no stale-model risk from a leftover loop variable.

**Workflow:** load → clean → encode (Label for binary columns, `pd.get_dummies(..., drop_first=True, dtype=int)` for `Property_Area`) → X, y → `train_test_split` (`random_state=42`, unchanged so results stay comparable) → baseline `cross_val_score` on the pipeline → `GridSearchCV` (`svm__C` in [0.1, 1, 10, 100], `svm__gamma` in [0.01, 0.1, 1], 12 combinations × 5 folds) → `grid.predict(X_test)` once → metrics.

### Results

**Baseline, default SVC in the pipeline, 5-fold CV on train:**
```
Fold scores: 0.848, 0.806, 0.827, 0.786, 0.806
Mean: 0.8146
```

**GridSearchCV:**
```
Best params: {'svm__C': 1, 'svm__gamma': 0.1}
Best CV score: 0.8146
```

**Test set, evaluated once:**
```
Confusion Matrix: [[18 25]
                    [ 1 79]]
Accuracy: 78.9%

              precision  recall  f1-score  support
0 (Rejected)      0.95    0.42      0.58       43
1 (Approved)      0.76    0.99      0.86       80
```

### Findings

**1. Tuning made no difference.** The baseline CV mean (0.8146) and the best tuned CV score (0.8146) are identical to 16 digits. The default SVC (`C=1`) was already as good as anything in the grid, so 60 fits found nothing the defaults didn't already have.

**2. SVM's test predictions are a single rule.** Comparing predictions against `Credit_History > 0.5` gave `(y_pred == rule_pred).mean() = 1.0`: all 123 test predictions match that one rule exactly. This is the second model, after the Decision Tree stump, that collapses to a one-feature rule on this dataset.

**3. `cv_results_` shows the whole shape of the search:**

| Region | Configs | CV score |
|---|---|---|
| Underfit | `C=0.1` (all three gammas), `C=1, gamma=0.01` | 0.6965 |
| Plateau (rank 1, four-way tie) | `C=10/γ=0.01`, `C=10/γ=0.1`, `C=1/γ=0.1`, `C=100/γ=0.01` | 0.8146 |
| Overfit | `C=100/γ=0.1` → 0.804, `C=100/γ=1` → 0.780 | declining |

Nothing rose above the plateau, and making the model more complex made CV score fall. `best_params_` reports `C=1, gamma=0.1` only because that is the first of the four tied configs in the grid's search order, not because it is meaningfully better than the others. The underfit score of 0.6965 looks like the "always predict Approved" rate; this was not separately verified (`y_train.mean()` would confirm it).

**4. No change versus the earlier SVM.** Test accuracy is 78.9% both times and the confusion matrix is identical, even though the earlier tuning chose `C=0.1` and this one chose `C=1`. Changing the tuning method changed the chosen parameters but not the answer.

### Comparison — what reaches 78.9% on this dataset

| Model | How tuned | Test accuracy | CV mean |
|---|---|---|---|
| SVM (`C=0.1`) | loop on test accuracy | 78.9% | — |
| SVM (`C=1, gamma=0.1`) | GridSearchCV in a Pipeline | 78.9% | 0.8146 |
| Decision Tree stump (`max_depth=1`) | GridSearchCV | 78.9% | 0.8146 |
| Rule: `Credit_History > 0.5` | none | 78.9% | — |

All four give the same test accuracy and the same confusion matrix. The test set has 123 rows, so accuracy noise is roughly ±3–4 points anyway; an identical score is not evidence of improvement, and here it is evidence of there being nothing more to find.

### Revisiting the 0.42 recall explanation

Earlier READMEs, and the comparison table above, attributed the identical 0.42 minority-class recall across algorithms to class imbalance (~69% / 31%) alone. The stump and SVM results suggest a likelier contributing cause: on this dataset the only strong signal is `Credit_History`, and the 25 rejected applicants these models miss mostly have `Credit_History = 1`, so nothing in the data separates them from approved ones. Imbalance is probably still a factor, but not the full story. This is supported by the stump, this SVM, and the Decision Tree revisit, and is not proven for every algorithm (AdaBoost's feature importance, for example, was much more spread out). It matters before reaching for `class_weight` or SMOTE, since neither can change what a single-feature split can see.

### Mistakes made in the revisit

1. **Skipped the baseline `cross_val_score` at first** — ran `pipe.fit(X_train, y_train)` instead, which trains once and prints nothing. Without a baseline there was no way to tell whether the grid improved anything; adding it is what revealed finding #1.
2. **Skipped `drop_first=True, dtype=int` in `pd.get_dummies`** on the first run, leaving a redundant `Property_Area` column that also inflates that feature's weight in RBF distance. Fixed on the second run.
3. **Filled `Loan_Amount_Term` and `Credit_History` with `.mean()` instead of `.mode()[0]`** — the latter is binary, so 50 rows got a fractional `0.842`. Left unchanged in this notebook. It did not alter the result here (0.842 > 0.5, so those rows still fall on the "approve" side of the rule), but it is the wrong choice for a binary column.
4. **Leftover clutter:** the useless `pipe.fit(X_train, y_train)` cell and an `if 'Property_Area' in df.columns` guard (a band-aid for re-running a cell without restarting the kernel) are both still in the notebook. Restart Kernel + Run All is the actual fix.

### Limitation

Imputation (median/mean/mode) and encoding are fitted on all 614 rows before the train-test split, so statistics from the test rows leak into preprocessing. Scaling is the only step moved inside the `Pipeline`. The effect is probably small here, but it is the same kind of leak the `Pipeline` was introduced to prevent.