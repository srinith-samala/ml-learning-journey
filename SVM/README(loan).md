# SVM (Support Vector Machine) Classifier

Loan Prediction dataset (same as all previous algorithms) — the final supervised classification algorithm in the core roadmap, bringing a margin-maximization paradigm distinct from equation-based, tree-based, distance-voting, probability-based, and boosting approaches already tried.

AI usage disclosure: Used Claude for concept explanations (maximum margin, support vectors, the kernel trick) and code review across two debugging rounds. Code was written and run independently.

## Pipeline followed

Import (`SVC` from `sklearn.svm`) → Load → Explore → Drop `Loan_ID`, `Married`, `Dependents` → Missing Values → Encoding (Label Encoding for binary columns + One-Hot for `Property_Area` — SVM calculates on raw numeric values like Logistic Regression/KNN, so One-Hot is needed again) → Define X, y → Train-Test Split → **Scaling (MinMax, mandatory)** → Baseline `SVC(kernel='rbf', C=1.0, probability=True)` → Tune `C` via loop → Retrain at best `C` → Metrics → Overfitting check.

## Key concepts

**Maximum margin:** instead of just finding *a* line that separates two classes, SVM finds the one with the widest possible gap ("margin") between itself and the closest points of each class — the idea being that a wider margin generalizes better to new data.

**Support vectors:** only the points closest to the boundary actually define it — points far from the boundary have zero influence on where the line sits. This is where the algorithm's name comes from.

**Kernel trick:** real data often isn't linearly separable (e.g. one class forming a ring around another). Rather than literally transforming data into a higher dimension (expensive), a kernel function (here, `rbf`) mathematically achieves the same effect — making the data separable in an implicit higher-dimensional space — without the explicit, costly transformation.

**`C` hyperparameter:** controls how strictly the margin is enforced. Small `C` allows more misclassified points for a wider, simpler margin (risk of underfitting); large `C` punishes misclassification harder, producing a narrower, more complex boundary (risk of overfitting). Tuned the same way as `max_depth`/`n_estimators` in earlier notebooks: loop over candidate values, pick whichever gives the best test accuracy.

**No `feature_importances_`:** unlike every tree-based model tried so far, SVM doesn't expose feature importances directly (only a rough `coef_`-based workaround exists, and only for a linear kernel) — a genuinely new limitation compared to Decision Tree/Random Forest/XGBoost/AdaBoost.

## Mistakes made (and fixed)

1. **Trained the first model on unscaled data** — built `Scaled_X_train`/`Scaled_X_test` correctly but then called `model.fit(X_train, y_train)` on the raw (unscaled) data by mistake. Since SVM is margin/distance-based, this produced a completely collapsed model that only ever predicted the majority class (`[[0,43],[0,80]]`, 65% accuracy, 0.00 precision/recall on class 0) — the same failure mode seen whenever scaling is skipped for a distance-based algorithm. Fixed by fitting on `Scaled_X_train` instead.

2. **Repeated the same mistake in the final Train/Test score check** — after correctly predicting on `Scaled_X_test`, the last two lines still called `model.score(X_train, y_train)` / `model.score(X_test, y_test)` on unscaled data (leftover from before the fix), silently producing a mismatched, lower, and meaningless score (0.696/0.650) even though the confusion matrix and accuracy above it were already correct. Fixed by scoring against `Scaled_X_train`/`Scaled_X_test` consistently.

## Choosing C

Looped `C` over `[0.1, 1, 10, 100]` (log-scaled values, plotted with `plt.xscale('log')` since the values aren't evenly spaced) and picked the best via `np.argmax(test_scores)`. Best value: **`C=0.1`** — very close to the default (1.0), so results barely changed from the baseline run.

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

## Comparison — all ten algorithms (Loan Prediction)

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

SVM tied Logistic Regression for the best accuracy on this dataset (78.9%), and landed on the now-familiar 0.42 minority-class recall shared by nine of ten algorithms tried — further confirming the dataset's class imbalance (~69%/31%), not model choice, as the real ceiling here. This closes out the core supervised classification roadmap (Linear/Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost, Gradient Boosting, AdaBoost, SVM).