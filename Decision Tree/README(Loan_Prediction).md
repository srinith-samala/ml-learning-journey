# Decision Tree Classifier

Loan Prediction dataset (same as Logistic Regression) — used to compare how a tree-based model handles the same data differently from a linear model.

AI usage disclosure: Used Claude for roadmap structuring, concept explanations, and debugging help, and again for concept explanations and code review during the Cross-Validation / GridSearchCV revisit at the end of this README. Code was written and run independently.

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

Both models land on the same 0.42 recall for the minority class (Rejected). The first explanation was the dataset's class imbalance (~69% approved); the revisit below (`model_tuning_dt.ipynb`) found a likelier contributing cause, so read that section before drawing a conclusion from this table.

---

## Revisited: Cross-Validation & GridSearchCV (`model_tuning_dt.ipynb`)

The `max_depth=4` above was picked by looping over depths and comparing **test** accuracy. That lets the test set influence the choice, so the reported score is best-of-N rather than an honest estimate. This revisit redoes the tuning inside the training data using Cross-Validation, and touches the test set once, at the very end.

### What was used

**Cross-Validation (K-Fold, `cv=5`):** the training data is split into 5 folds. The model trains 5 times, each time holding out a different fold for validation, and the 5 scores are averaged. The test set is never involved, and the average is more stable than a single split's score.

**GridSearchCV:** automates the loop over hyperparameter combinations with CV built in. Each combination is scored by 5-fold CV on `X_train`, and the best one is refit on all of `X_train` (`refit=True`), so `grid.predict()` uses the best model directly. This also removes the stale-model bug from the loops above, where the loop's last leftover model stayed in use.

**Workflow:** load → clean → encode → X, y → `train_test_split` (`random_state=42`, unchanged so results stay comparable) → baseline `cross_val_score` → `GridSearchCV` on `X_train` only → `grid.predict(X_test)` once → metrics.

### Results

**Baseline, untuned tree, 5-fold CV on train:**
```
Fold scores: 0.747, 0.694, 0.684, 0.735, 0.786
Mean: 0.729
```
The folds range from 0.684 to 0.786, a 10-point spread for the same model. This is the practical reason a single score from a 123-row test set is noisy (roughly ±3–4 points).

**GridSearchCV** (`max_depth` in [1–5], `min_samples_leaf` in [1, 5, 10, 20, 30, 50], 30 combinations × 5 folds):
```
Best params: {'max_depth': 1, 'min_samples_leaf': 1}
Best CV score: 0.8146
```
Several configurations tie at 0.8146 and `GridSearchCV` returns the first in its search order, which is why the shallowest one won. An earlier, narrower grid (depth 2–5) returned `max_depth=2, min_samples_leaf=10` with the identical score of 0.8146, which confirms the tie.

**Test set, evaluated once:**
```
Confusion Matrix: [[18 25]
                    [ 1 79]]
Accuracy: 78.9%

              precision  recall  f1-score  support
0 (Rejected)      0.95    0.42      0.58       43
1 (Approved)      0.76    0.99      0.86       80
```

### Main finding: the best model is a single rule

```
Credit_History: 1.0   (every other feature: 0.0)
```

With `max_depth=1` the tree is a stump with one split on `Credit_History`: approve if it is 1, reject otherwise. That one rule scores 78.9% on the test set, equal to Logistic Regression and SVM and above Random Forest, XGBoost, KNN, Gradient Boosting, and AdaBoost on this dataset.

### Comparison — Decision Tree versions

| Version | How it was chosen | Accuracy |
|---|---|---|
| Unrestricted tree | no tuning | 70.7% (test) / 0.729 (CV mean) |
| `max_depth=4` | manual loop on test accuracy | 77.2% (test) |
| `max_depth=1` stump | GridSearchCV, 5-fold CV | 78.9% (test) / 0.8146 (CV) |

Tuned versus untuned is a real improvement (about 8 points, visible in both CV and test). Manual `max_depth=4` versus GridSearchCV is **not** a meaningful difference: 77.2% vs 78.9% is about 2 rows out of 123, well inside test-set noise. The CV-based result is simply the more trustworthy way to reach a tuned model.

### Revisiting the 0.42 recall explanation

The comparison table above, and earlier READMEs in this repo, attributed the identical 0.42 minority-class recall across algorithms to class imbalance (~69% / 31%) alone. The stump result suggests a likelier contributing cause: on this dataset the only strong signal is `Credit_History`, and the 25 rejected applicants the models miss mostly have `Credit_History = 1`, so nothing in the data separates them from approved ones. Imbalance is probably still a factor, but not the full story. This is a hypothesis supported by the stump and this tree's results, not proven for every algorithm (AdaBoost's feature importance, for example, was much more spread out). Worth keeping in mind before applying `class_weight` or SMOTE, since neither can change what a single-feature split can see.

### Mistakes made in the revisit (and fixed)

1. **`Loan_Amount_Term` missing values left unfilled** (14 rows), the same miss as in the Random Forest notebook. A tree tolerates it, but SVC would crash on it.
2. **Accepted the grid's edge values without checking.** The first grid ([2–5] × [1, 5, 10]) put its best values on the edge (`max_depth=2` the smallest, `min_samples_leaf=10` the largest tried), a sign the true best might lie outside the grid. Widening it showed the optimum is a tie, not a boundary effect.
3. **Fitted the grid twice** (`grid.fit` run again for no reason, repeating 60 fits).
4. **Skipped the baseline `cross_val_score` at first**, so there was no "before tuning" CV number to compare against.
5. **Cells out of order** (`grid.best_score_` placed before `grid.fit`) and edited without a fresh Run All, so stale outputs from an old run were mistaken for current results. The corrected, fresh run changed the reported best params from `max_depth=2` to `max_depth=1`.

`RuntimeWarning: invalid value encountered in cast` appears during `grid.fit`. Results are unaffected; it comes from numpy handling masked arrays in `GridSearchCV`'s result table and can be ignored.