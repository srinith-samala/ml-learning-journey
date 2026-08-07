# Linear Regression — KC House Sales Prediction

End-to-end Linear Regression on the King County (Seattle) house sales dataset (`kc_house_data.csv`, ~21.6k rows).

AI usage disclosure: Used Claude for roadmap structuring and code review. Code was written and run independently.

## Pipeline followed

Import → Load Dataset → Explore (`head`, `info`, `isnull`, `duplicated`) → Define X, y → Train-Test Split → Create Model → Train Model → Predict → Evaluate (MAE, MSE, RMSE, R²) → Actual vs Predicted Scatter Plot

## Results

```
MAE:  127,474.10
MSE:  45,164,817,780.90
RMSE: 212,520.16
R²:   0.7012
```

## ⚠️ Known issues (not yet fixed)

Flagged during review, left as-is for now — will apply these fixes going forward starting with Decision Tree:

1. **Execution order issue** — the `date` column was dropped (Cell 9) *after* `model.fit()` was already called (Cell 8). Since `date` is a text column, this only worked because the notebook wasn't run in a single clean top-to-bottom pass — the displayed cell order doesn't match the actual execution order. Needs a kernel restart + Run All to get a trustworthy result.

2. **`id` column not dropped** — pure identifier, no predictive value, adds noise as a feature.

3. **`zipcode` column used as raw numeric** — it's actually categorical (a location code), not a continuous number. Feeding it to Linear Regression as-is implies a false numeric relationship between zip codes (same issue as using Label Encoding on a nominal column). Should be dropped or properly encoded.

Because of point 1 especially, the R² (0.70) and error metrics above should be treated as provisional, not final — they may change once the notebook is run cleanly with the `date`/`id`/`zipcode` fixes applied.

## Next steps

- Fix execution order, drop `id`, handle `zipcode` properly (drop or encode)
- Re-run clean (Kernel Restart → Run All) and update these metrics