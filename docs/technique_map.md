# Technique map (requirement R2)

Use this table for the **technique map slide**. Every row points to the file that implements it, so it matches the code.

| Area | Technique (from the homework list) | Component in the app | Where in the code |
|---|---|---|---|
| **A · Neural networks** | Feed-forward network trained with **SGD + backpropagation** | Model 3: 200 → 128 (ReLU) → 2 classifier | `src/w2v_ffn.py` → `FeedForward`, `torch.optim.SGD`, `loss.backward()` |
| A | **Dropout** (0.3) and early stopping on validation macro-F1 | Model 3 | `src/w2v_ffn.py` |
| **B · Classical NLP** | **Regular expressions** for text normalization (URLs, emoji, hashtags, leetspeak, `f.u.c.k`, repeated letters, accents and Turkish letters) | Shared preprocessing for every model | `src/normalize.py` |
| B | **Minimum edit distance** (Damerau-Levenshtein, dynamic programming) against a bad-word **lexicon** | Model 1: rule baseline | `src/lexicon.py` → `edit_distance`, `LexiconMatcher` |
| B | **Tokenization** (word tokens) | Models 1-3 | `src/normalize.py` → `tokenize` |
| B | **TF-IDF** (word 1-2-grams) + logistic regression | Model 2: classical baseline | `src/tfidf_model.py` |
| B | **word2vec** (skip-gram, 100-d) trained by us on the training messages | Model 3: input features | `src/w2v_ffn.py` → `train_word2vec` |
| **C · Transformers** | **Fine-tuning BERT** (bert-base-uncased) with a classification head on [CLS] | Model 4 | `src/bert_model.py` |
| C | **Subword tokenizer** (WordPiece) and multi-head **self-attention** encoder | Models 4-5 | `AutoTokenizer` / `AutoModelForSequenceClassification` |
| **D · Efficient models** | **Quantization (dynamic INT8)** with ONNX Runtime, after exporting to **ONNX** | Model 5: the model the app runs by default | `src/quantize.py` → `export_onnx`, `quantize_onnx` |
| D | Dynamic INT8 with PyTorch (`quantize_dynamic` on `nn.Linear`), extra comparison row | Model 5b | `src/quantize.py` → `quantize_torch` |
| D | **Benchmarking** size, latency and accuracy against a **latency budget** (50 ms / message on CPU) | Results table, app side panel | `src/evaluate.py`, `app.py` |

## Required components of Use Case 4, and where they are

| Homework says | We do |
|---|---|
| B: regex normalization of noisy text (leetspeak, repeated letters, Turkish characters) | `src/normalize.py` |
| B: an edit-distance lexicon matcher as a rule baseline | `src/lexicon.py` (Model 1) |
| B: a tf-idf baseline | `src/tfidf_model.py` (Model 2) |
| A: your own word2vec trained on the training messages, feeding a feed-forward network | `src/w2v_ffn.py` (Model 3) |
| C: BERTurk or a multilingual BERT fine-tuned for classification | `src/bert_model.py` (Model 4). We use **English** data (the homework allows "Turkish (or English)"), so the matching English model is `bert-base-uncased`. |
| D: a latency budget you set, met with quantization | 50 ms budget in `config.py`. Measured in `src/evaluate.py` and shown live in `app.py` (Model 5). |
