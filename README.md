# ML Learning Journey

Personal practice repo for learning ML algorithms from scratch — implementation, debugging, and notes on what went wrong along the way. This is a learning log, not a polished project repo.

AI usage disclosure: Used Claude for roadmap structuring, code review, and debugging help throughout. Code was written and run independently; Claude reviewed output and flagged bugs.

## Progress

- [x] Linear Regression
- [x] Logistic Regression (+ Encoding)
- [x] Decision Tree
- [x] Random Forest
- [x] KNN
- [x] Naive Bayes
- [x] XGBoost
- [x] Gradient Boosting
- [ ] AdaBoost
- [ ] SVM
- [ ] Ridge / Lasso / ElasticNet
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

Two datasets: Loan Prediction and Iris (first multiclass problem — 3 species, all-numeric features).

**Key concept:** no real "training" happens — for every prediction, KNN calculates Euclidean distance to all training points and takes a majority vote among the `K` nearest ones. Because it's distance-based: scaling is mandatory again (like Logistic Regression), and One-Hot Encoding is back for nominal columns (unlike the tree models, which only needed Label Encoding).

**Choosing K:** looped `n_neighbors` from 1–20 and picked the value with best test accuracy (K=9 on Loan Prediction), rather than guessing a number upfront — same tuning discipline as `max_depth` for Decision Tree.

**Iris result:** 100% accuracy at every K value — a legitimate result (not a red flag) since Iris's 3 species are famously cleanly separable, unlike real-world messier data. First 3×3 confusion matrix.

**Cross-algorithm insight:** every classification algorithm tried on Loan Prediction so far shows the *exact same* 0.42 recall on the minority class (Rejected). Strong evidence the bottleneck is the dataset's class imbalance (~69%/31%), not the choice of algorithm — flagged for later with `class_weight='balanced'` or resampling techniques.

## 06 — Naive Bayes

Three datasets: Loan Prediction, Diabetes (Pima Indians, continuous numeric features — GaussianNB's ideal setting), and SMS Spam Collection (first text classification problem).

**Key concepts:** Bayes Theorem (`P(A|B) = [P(B|A)×P(A)] / P(B)`) — worked through a rare-disease-test example showing why a positive result on a rare condition can still mean low actual probability, since the base rate matters as much as test accuracy. The "naive" independence assumption treats all features as unrelated (rarely true in reality) purely to make the probability calculation fast — a simplification that still works well in practice because relative class ranking tends to stay correct even when exact probabilities are off.

**GaussianNB vs MultinomialNB:** GaussianNB assumes continuous, normally-distributed features (used for Loan Prediction and Diabetes); MultinomialNB is built for discrete count data and is the standard choice for text (used for SMS Spam).

**Text vectorization (new concept):** raw text can't use LabelEncoder/OneHotEncoder — instead `TfidfVectorizer` turns each unique word across the dataset into its own feature/column, weighted by how distinctive that word is to a subset of messages (vs common filler words).

**Results:** Diabetes gave the best-balanced recall yet (0.69 on minority class) — the most "natural" fit for GaussianNB. SMS Spam gave the best accuracy of any algorithm/dataset combination so far (96.7%), with perfect spam precision (1.00) — direct confirmation that Naive Bayes' real strength is high-dimensional text classification, where distance-based models like KNN would struggle.

## 07 — XGBoost

Loan Prediction dataset — first Boosting-family algorithm (`xgboost` library, not sklearn).

**Key concept — Boosting vs Bagging:** Random Forest (Bagging) builds trees independently and averages them. XGBoost (Boosting) builds trees *sequentially* — each new tree is trained to predict the residual (error) of all trees before it, and predictions are updated as `old_prediction + (learning_rate × new_tree_prediction)`. This mirrors Gradient Descent: taking small, scaled steps that reduce error over many iterations rather than jumping straight to a solution.

**Result:** 76.4% accuracy — first algorithm to nudge the minority-class recall above the 0.42 every other model landed on (0.44), consistent with Boosting's sequential error-focus. Still not the top performer on this small dataset (Logistic Regression remains ahead at 78.9%) — XGBoost's advantage tends to show up more on larger, more complex data.

## 08 — Gradient Boosting

Loan Prediction dataset, identical pipeline and hyperparameters to the XGBoost run, for direct comparison — sklearn's `GradientBoostingClassifier` is the "unoptimized ancestor" of XGBoost: same core sequential-residual algorithm, but without XGBoost's added regularization, automatic missing-value handling, and internal parallelization.

**Result:** 74.8% accuracy, and a noticeably larger train/test gap (0.892 vs 0.748) than XGBoost's run — a clean, practical confirmation that XGBoost's engineering on top of the same base algorithm meaningfully reduces overfitting and improves accuracy, even with identical `n_estimators`/`learning_rate`/`max_depth`. Lowest-scoring model on this dataset so far — extra model complexity doesn't pay off on a small, single-feature-dominated dataset like this one.