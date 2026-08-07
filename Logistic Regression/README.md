# Logistic Regression + Encoding

Two datasets used: Titanic (intro, all-numeric-friendly via seaborn) and Loan Prediction (Analytics Vidhya/Kaggle) — the second one specifically to practice encoding on real categorical data.

AI usage disclosure: Used Claude for roadmap structuring, code review, and debugging help. Code was written and run independently; Claude reviewed output and flagged bugs.

## Pipeline followed

Import → Load → Explore → Handle Missing Values → Feature Selection → Encoding → Define X, y → Train-Test Split → Scaling (MinMax) → Create Model → Train Model → Predict → Predict Probability → Confusion Matrix → Accuracy → Classification Report

## Encoding covered

- **Label Encoding** — for binary/ordinal columns (Gender, Self_Employed, Education, Loan_Status)
- **One-Hot Encoding** (`pd.get_dummies`, `drop_first=True`) — for nominal columns with no natural order (Property_Area)

## Mistakes I made (and fixed)

1. **Self-concat bug** — accidentally ran `pd.concat([df, df], axis=1)` twice, duplicating every column 4x. Broke scaling and model training silently; model ended up predicting only the majority class. Root cause of a corrupted run: 0.65 accuracy with 0.00 precision/recall on the minority class.

2. **Dropped `Credit_History` without checking predictive value first** — turned out to be the strongest predictor in the Loan dataset. Dropping it made the model default to predicting the majority class almost every time (accuracy 65%, class 0 recall = 0.00). Adding it back (with missing values imputed via mean) raised accuracy to ~79% and gave the model actual signal to detect the minority class (class 0 recall went from 0 to 0.42).

3. **Reused one `LabelEncoder()` object across multiple columns** — `le.classes_` only retains the last-fit column's mapping, so I lost the ability to check what 0/1 meant for earlier-encoded columns (Gender, Education, Self_Employed). Fixed by creating a separate encoder per column (`le_gender`, `le_education`, `le_self_employee`, `le_loan_status`).

4. **Didn't restart kernel after fixing bugs** — kept re-running edited cells without restarting, which left stale variables in memory and produced misleading "still broken" outputs even after the actual code was fixed. New habit: fix code → Restart Kernel → Run All → re-check output, every time.

## Final results — Loan Prediction (after fixes)

```
Confusion Matrix: [[18 25]
                    [1 79]]
Accuracy: 78.9%

              precision  recall  f1-score  support
0 (Rejected)      0.95    0.42      0.58       43
1 (Approved)      0.76    0.99      0.86       80
```

**Open observation:** class 0 recall (0.42) is still much lower than class 1 recall (0.99) — model leans heavily toward predicting approval. Likely driven by natural class imbalance in the dataset (~69% approved / 31% rejected). Worth revisiting with `class_weight='balanced'`, or comparing against tree-based models later to see if they handle the imbalance better by default.