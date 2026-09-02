# Naive Bayes — SMS Spam Detection (Text Classification)

Third Naive Bayes dataset, done independently. SMS Spam Collection Dataset (Kaggle/uciml, 5572 messages) — the first text classification problem tried, chosen specifically to see Naive Bayes' real strength on high-dimensional text data.

AI usage disclosure: Used Claude for a file-encoding fix (`UnicodeDecodeError`) and a code review explaining the vectorization choices made. Core implementation — including choosing `TfidfVectorizer` and `MultinomialNB` — was done independently.

## Pipeline followed

Import (including `CountVectorizer`, `TfidfVectorizer` from `sklearn.feature_extraction.text`, `MultinomialNB` from `sklearn.naive_bayes`) → Load (`encoding='latin-1'` — file wasn't UTF-8) → Explore → Drop unnamed empty columns, keep only `v1`/`v2`, rename to `label`/`message` → Encode target (`ham`→0, `spam`→1 via `.map()`) → Define X (raw text column, no other features) and y → Train-Test Split → **Text Vectorization** (TF-IDF) → Create Model (`MultinomialNB`) → Train → Predict → Confusion Matrix → Accuracy → Classification Report

## Key new concept: Text Vectorization

Traditional encoding (`LabelEncoder`, `OneHotEncoder`) doesn't work here — each row is a full sentence, not a fixed category. Instead, the text column gets converted into a huge sparse matrix where every unique word across the dataset becomes its own column ("feature"), and each message becomes a row of numbers representing that word's presence/frequency.

**`CountVectorizer`** — just counts how many times each word appears in each message.

**`TfidfVectorizer`** — weighted version. Down-weights words that appear in almost every message (e.g. "the", "is") and up-weights words that are more distinctive to a smaller set of messages (e.g. "free", "winner" showing up mostly in spam). Ended up using this over CountVectorizer for the final model.

**`stop_words='english'`** — strips out common English filler words before vectorizing, since they carry no classification signal.

## Why `MultinomialNB` instead of `GaussianNB`

`GaussianNB` (used in the Loan Prediction and Diabetes notebooks) assumes continuous, roughly normally-distributed features. Word counts/TF-IDF scores are **discrete count-like data**, not continuous — `MultinomialNB` is the variant built specifically for this kind of data, and it's the standard choice for text classification with Naive Bayes.

## Fix applied

`UnicodeDecodeError` on `pd.read_csv('spam.csv')` — the file isn't UTF-8 encoded (SMS text can include special characters that don't map cleanly to UTF-8). Fixed with `pd.read_csv('spam.csv', encoding='latin-1')`.

## Cleanup note

Built both a `CountVectorizer` and a `TfidfVectorizer` version using the same variable name (`X_train_cv`) — the TF-IDF cell ran second and silently overwrote the CountVectorizer result, so the final model only ever trained on TF-IDF vectors. Not a bug (the result is valid), but redundant/confusing code — the CountVectorizer cell should be removed or clearly marked as an alternative not used in the final run.

## Results

```
Confusion Matrix: [[965   0]
                    [ 37 113]]
Accuracy: 96.7%

              precision  recall  f1-score  support
0 (Ham)           0.96    1.00      0.98      965
1 (Spam)          1.00    0.75      0.86      150
```

Best accuracy of any algorithm/dataset combination so far, and precision on Spam is perfect (1.00) — every message flagged as spam actually was spam, meaning zero legitimate messages got wrongly blocked. Recall on Spam (0.75) means about a quarter of actual spam still slips through as "Ham," which is the realistic trade-off — this dataset is fairly imbalanced (4825 Ham vs 747 Spam, ~87%/13%), and the model is deliberately cautious about flagging real messages as spam given `stop_words` removes fewer of the most spam-distinctive words.

## Why this dataset (unlike Loan Prediction / Diabetes) shows Naive Bayes' actual strength

The feature space here is enormous — thousands of unique words, each its own dimension — which is exactly the setting Naive Bayes was designed for. Distance-based models like KNN would be extremely slow and less effective at this dimensionality (curse of dimensionality), while Naive Bayes' simple per-word probability multiplication scales easily regardless of how many word-features exist.

## Naive Bayes across all three datasets

| Dataset | Type | Accuracy | Notes |
|---|---|---|---|
| Loan Prediction | Numeric + categorical | 78.0% | Bottlenecked by class imbalance |
| Diabetes | Continuous numeric | 76.6% | Best-balanced recall (GaussianNB's ideal setting) |
| SMS Spam | Text (thousands of features) | **96.7%** | Best overall — Naive Bayes' actual home turf |