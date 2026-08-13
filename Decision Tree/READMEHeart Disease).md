# Decision Tree — Heart Disease Prediction (Practice Dataset)

Second Decision Tree dataset, done independently to practice the full pipeline: encoding, depth tuning, metrics, and tree visualization on new data (UCI/Kaggle Heart Failure Prediction Dataset, 918 rows).

AI usage disclosure: Used Claude for debugging help and code review. Code was written and run independently.

## Pipeline followed

Same as the Loan Prediction Decision Tree: Import → Load → Explore → Missing Values (none found here) → Encoding (Label Encoding, separate encoder per column) → Define X, y → Train-Test Split → Loop across `max_depth` values to compare train/test accuracy → Retrain final model at best depth → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report → Overfitting Check → Feature Importance → Tree Visualization

## Choosing `max_depth`

Looped `max_depth` from 2 to 10, plotted train vs test accuracy for each. Test accuracy peaked at **`max_depth=2`** and declined as depth increased — train accuracy kept climbing toward 1.0 the deeper the tree went, the classic overfitting pattern. Picked the depth where test accuracy was highest, not where train accuracy was highest.

## Mistakes I made (and fixed)

1. **Reused the loop's leftover `model` variable instead of retraining at the chosen depth** — after the depth-comparison loop, `model` still held the last iteration (`max_depth=10`, overfit). Needed to explicitly rebuild and refit the model at `best_depth` before running predictions/metrics, or every downstream result reflects the wrong (overfit) model.

2. **Typo — `Y_train` instead of `y_train`** — Python is case-sensitive; this silently raised `NameError` and caused the retraining step to fail, so the stale overfit model from mistake #1 kept getting used through the rest of the notebook without any error showing up in later cells.

3. **Leftover `class_names=['Rejected', 'Approved']` in `plot_tree()`** — copy-pasted from the Loan Prediction notebook; this dataset's classes are `['Normal', 'Heart Disease']`.

4. **Encoder reuse bug (same class of mistake as before)** — accidentally reused `le_sex.fit_transform()` for `RestingECG`, `ExerciseAngina`, and `ST_Slope` instead of their own encoder objects. Encoded values were still correct (each `fit_transform` call recomputes independently), but `le_sex.classes_` ended up showing the wrong column's categories. Fixed by using each column's own encoder object consistently.

## Results — before vs after fixing the stale-model bug

| | Buggy (stale depth=10 model) | Fixed (genuine depth=2) |
|---|---|---|
| Train Score | 0.978 | 0.834 |
| Test Accuracy | 0.793 | **0.842** |

Fixing the bug both removed the overfitting (train/test scores now close together) and improved actual test performance.

## Final results (max_depth=2)

```
Confusion Matrix: [[61 16]
                    [13 94]]
Accuracy: 84.2%

              precision  recall  f1-score  support
0 (Normal)        0.82    0.79      0.81       77
1 (Heart Disease) 0.85    0.88      0.87      107
```

Balanced recall across both classes (0.79 and 0.88) — better balance than the Loan Prediction model, which struggled much more on its minority class.

## Feature Importance

```
ST_Slope        0.809
Cholesterol     0.105
ChestPainType   0.086
(all others)    0.000
```

`ST_Slope` dominates — with `max_depth=2`, the tree only had room to use its top 3 features, everything else got 0 importance. Same pattern as `Credit_History` dominating the Loan Prediction tree.

## Comparison — Loan Prediction vs Heart Disease Decision Trees

| Metric | Loan Prediction (max_depth=4) | Heart Disease (max_depth=2) |
|---|---|---|
| Test Accuracy | 77.2% | 84.2% |
| Train/Test Gap | 0.833 vs 0.772 | 0.834 vs 0.842 (test even higher) |
| Minority Class Recall | 0.42 | 0.79 |

Heart Disease dataset generalized noticeably better at a much shallower depth — likely a cleaner/more separable dataset than Loan Prediction, which has a stronger class imbalance problem.