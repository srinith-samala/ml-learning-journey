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
- [x] AdaBoost
- [x] SVM
- [ ] Ridge / Lasso / ElasticNet
- [ ] K-Means
- [ ] PCA

**Core supervised classification roadmap complete** (Linear Regression → Logistic Regression → Decision Tree → Random Forest → KNN → Naive Bayes → XGBoost → Gradient Boosting → AdaBoost → SVM). Next: regression variants, then unsupervised learning (K-Means, PCA).

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

## 09 — AdaBoost

Two datasets: Loan Prediction and Telco Customer Churn — the third and final Boosting-family algorithm, genuinely different in mechanism from the other two.

**Key concept — AdaBoost vs Gradient Boosting:** Gradient Boosting/XGBoost fit each new tree to the previous *residual* (the target changes each round, data stays the same). AdaBoost does the opposite — the target never changes, but **sample weights** do. Every sample starts with equal weight; after each weak learner (a "stump" — `max_depth=1` tree), misclassified samples get their weight increased so the next learner is forced to focus on the hard cases. Each learner's vote is scaled by `α = ½·ln((1−error)/error)` — a learner no better than random gets zero vote, worse-than-random gets an inverted vote.

**Debugging theme:** this notebook needed several rounds of fixes — passing a `range()` object directly as `n_estimators` instead of looping over it, hardcoding a guessed `n_estimators` value instead of reading it off the tuning loop's results (`np.argmax`), and forgetting to re-run metrics on the retrained best-value model after fixing the guess.

**Results:** Loan Prediction (`n_estimators=10`) → 78.0% accuracy, the best-performing and best-generalizing Boosting algorithm on this dataset (smallest train/test gap of the three). Telco Churn (`n_estimators=130`) → 81.5% accuracy, 0.57 minority-class recall — the best result on that dataset across both ensemble methods tried, outperforming Random Forest (79.6%, 0.47 recall). AdaBoost's feature importance concentrated much more heavily into the top 2–3 features than Random Forest's spread-out importance, consistent with its weighted-voting mechanism repeatedly favoring the strongest predictors.

## 10 — SVM (Support Vector Machine)

Three datasets: Loan Prediction, and an independently-chosen Cricket Player Performance dataset (20,000 rows, synthetic, 4-class target `player_form_label`) — the final algorithm in the core supervised classification roadmap, and the first with a genuinely new paradigm (margin maximization) since KNN.

**Key concepts:** finds the decision boundary with the *maximum margin* between classes — only the closest points ("support vectors") define it; everything else has zero influence. The Kernel Trick (commonly `rbf`) lets it separate non-linearly-separable data without literally transforming it into a higher dimension. `C` controls margin strictness (small = wider/more tolerant, large = narrower/more aggressive, risking overfitting). No `feature_importances_` available — a first for this project, since every prior model (tree-based or not) exposed some form of feature ranking.

**Debugging theme:** trained the first model on unscaled data despite having built the scaled version (`Scaled_X_train`) — produced the same majority-class collapse seen whenever scaling is skipped for a distance-based algorithm (0.00 precision/recall on the minority class). A second, more subtle version of the same mistake reappeared in the final train/test score check, scoring against unscaled data even after predictions were correctly made on scaled data.

**Results:** Loan Prediction (`C=0.1`) → 78.9% accuracy, tying Logistic Regression for the best result on that dataset, with the same 0.42 minority recall shared by 9 of 10 algorithms tried — closing out strong, repeated confirmation that the dataset's class imbalance is the real ceiling. Cricket dataset (`C=100`, tuned) → 94.2% accuracy on a 4-class problem, but with a flagged concern: train score hit a perfect 1.0, indicating the tuned `C` value, while producing the best test accuracy, is also overfitting — a trade-off worth noting rather than treating the tuning result as purely "correct."

### Mistakes made across the project, now recurring enough to treat as a checklist

- Stale kernel state after edits (fix: Restart Kernel + Run All, every time)
- Reusing one `LabelEncoder()` across multiple columns (fix: one encoder object per column)
- `train_test_split()` return order mistakes (always `X_train, X_test, y_train, y_test`)
- Missing `()` on `.mean()`/`.median()`/`.mode()` silently corrupting a column's dtype
- Forgetting to retrain a final model at the best tuned hyperparameter after a loop, leaving a stale model in use for downstream metrics
- Not checking a dataset for disguised missing values (e.g. `0` standing in for missing where `0` is impossible) or data leakage (target-derived columns like `fantasy_points`/`player_rating` left in the feature set)