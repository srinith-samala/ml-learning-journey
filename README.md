# ML Learning Journey

Personal practice repo for learning ML algorithms from scratch — implementation, debugging, and notes on what went wrong along the way. This is a learning log, not a polished project repo.

AI usage disclosure: Used Claude for roadmap structuring, code review, and debugging help throughout. Code was written and run independently; Claude reviewed output and flagged bugs.

## Progress

- [x] Linear Regression
- [x] Logistic Regression (+ Encoding)
- [x] Decision Tree
- [ ] Random Forest
- [ ] KNN
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
