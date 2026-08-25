# ML Learning Journey

Personal practice repo for learning ML algorithms from scratch — implementation, debugging, and notes on what went wrong along the way. This is a learning log, not a polished project repo.

AI usage disclosure: Used Claude for roadmap structuring, code review, and debugging help throughout. Code was written and run independently; Claude reviewed output and flagged bugs.

## Progress

- [x] Linear Regression
- [x] Logistic Regression (+ Encoding)
- [x] Decision Tree
- [x] Random Forest
- [x] KNN
- [ ] Naive Bayes
- [ ] Gradient Boosting
- [ ] K-Means
- [ ] PCA

---

## 01 — Linear Regression

Standard end-to-end pipeline: load → explore → split → fit → predict → evaluate (MAE, MSE, RMSE, R²) → residual analysis.

## 02 — Logistic Regression (+ Encoding)

Two datasets: Titanic (sklearn-style intro) and Loan Prediction (Analytics Vidhya/Kaggle) — the second one specifically to practice encoding on real categorical data.

**Pipeline:** Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding → Define X, y → Train-Test Split → Scaling (MinMax) → Model → Train → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report

**Encoding covered:**
- Label Encoding — binary/ordinal columns (Gender, Self_Employed, Education, Loan_Status)
- One-Hot Encoding (`pd.get_dummies`, `drop_first=True`) — nominal columns (Property_Area)

### Mistakes I made (and fixed)

1. **Self-concat bug** — accidentally ran `pd.concat([df, df], axis=1)` twice, duplicating every column 4x. Broke scaling and model training silently; model ended up predicting only the majority class. Root cause of a corrupted 0.65 accuracy run with 0.00 precision/recall on the minority class.

2. **Dropped `Credit_History` without checking predictive value** — this turned out to be the strongest predictor in the Loan dataset. Dropping it made the model default to predicting the majority class almost every time (accuracy 65%, class 0 recall = 0.00). Adding it back (with missing values imputed) raised accuracy to ~79% and gave the model actual signal to detect the minority class (recall 0.42 on class 0, up from 0).

3. **Reused one `LabelEncoder()` object across multiple columns** — `le.classes_` only holds the last-fit column's mapping, so I lost the ability to check what 0/1 meant for earlier-encoded columns. Fixed by creating a separate encoder per column (`le_gender`, `le_education`, etc.).

4. **Didn't restart kernel after fixing bugs** — kept re-running edited cells without restarting, which left stale variables in memory and produced misleading "still broken" outputs even after the actual code was fixed. Now: fix → Restart Kernel → Run All → re-check output, every time.

### Final results (Loan Prediction, after fixes)

```
Confusion Matrix: [[18 25]
                    [1 79]]
Accuracy: 78.9%

              precision  recall  f1-score  support
0 (Rejected)      0.95    0.42      0.58       43
1 (Approved)      0.76    0.99      0.86       80
```

**Open observation:** class 0 recall (0.42) is still much lower than class 1 recall (0.99) — model leans heavily toward predicting approval, likely due to natural class imbalance in the dataset (~69% approved). Something to revisit with `class_weight='balanced'` or when comparing against tree-based models later.

## 03 — Decision Tree

Two datasets: Loan Prediction (same as above, for direct comparison) and Heart Disease (UCI/Kaggle Heart Failure Prediction).

**Pipeline:** Same as above, but Label Encoding only (no One-Hot needed — trees split on comparisons, not weighted sums), no scaling (not distance-based), plus `max_depth` tuning, feature importance, and tree visualization.

**Key lesson — choosing `max_depth`:** looped through depth values and plotted train vs test accuracy. The best depth is wherever *test* accuracy peaks, not train accuracy — train accuracy climbs toward 1.0 the deeper the tree goes (overfitting), while test accuracy peaks then declines. On Loan Prediction, `max_depth=4` was best; on Heart Disease, `max_depth=2` was best.

**Biggest mistake:** after running the depth-comparison loop, kept using the loop's leftover `model` variable (last depth tried) instead of explicitly retraining at the best depth — meant later metrics reflected an overfit model, not the tuned one. Also had a stale-model bug caused by a typo (`Y_train` vs `y_train`) silently failing and leaving old results in place.

**Feature importance:** `Credit_History` dominated the Loan tree (~77%); `ST_Slope` dominated the Heart Disease tree (~81%) — both datasets have one very strong predictor.

## 04 — Random Forest

Two datasets: Loan Prediction and Telco Customer Churn (Kaggle, 7043 rows — much larger, many more categorical columns).

**Key concepts:** Bagging (each tree trains on a random bootstrap sample of rows) + feature randomness (each split only considers a random subset of columns) — together these force trees to be genuinely different, so averaging their votes cancels out individual overfitting. `max_depth` matters far less here than for a single tree.

**Result:** on Loan Prediction, Random Forest (75.6%) actually underperformed Logistic Regression (78.9%) and Decision Tree (77.2%) — a good reminder that a more complex model isn't automatically better, especially on small datasets dominated by one strong feature. On the larger Telco Churn dataset, Random Forest hit 79.6% accuracy, and feature importance was spread across many features instead of one dominant column — consistent with feature randomness at work.

**Mistake:** forgot to fill missing values in `Loan_Amount_Term` before training — didn't error out (this sklearn version tolerates some missing values in tree models) but was still wrong practice; fixed and accuracy improved slightly.

## 05 — KNN (K-Nearest Neighbors)

Loan Prediction dataset — first distance-based algorithm tried, a different paradigm from equation-based (Linear/Logistic) and tree-based (Decision Tree/Random Forest) models.

**Key concept:** no real "training" happens — for every prediction, KNN calculates Euclidean distance to all training points and takes a majority vote among the `K` nearest ones. Because it's distance-based: scaling is mandatory again (like Logistic Regression), and One-Hot Encoding is back for nominal columns (unlike the tree models, which only needed Label Encoding).

**Choosing K:** looped `n_neighbors` from 1–20 and picked the value with best test accuracy (K=9), rather than guessing a number upfront — same tuning discipline as `max_depth` for Decision Tree.

**Cross-algorithm insight:** all four algorithms tried so far (Logistic Regression, Decision Tree, Random Forest, KNN) show the *exact same* 0.42 recall on the minority class (Rejected) on the Loan Prediction dataset. Strong evidence the bottleneck is the dataset's class imbalance (~69%/31%), not the choice of algorithm — flagged for later with `class_weight='balanced'` or resampling techniques.