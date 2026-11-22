# How it works: a study guide

The homework says you must be able to **explain any component** if the instructor asks. This guide goes through the system in the order a message travels, then answers the questions you are most likely to get.

---

## 0. The big picture

```
message ──> normalize (regex) ──> one of 5 models ──> P(clean), P(offensive) ──> label + confidence ──> mask? + latency log
```
- **Every model gets the same normalized text** and returns two probabilities. The app shows the bigger one as the label, and its probability as the confidence.
- All models implement the same small interface (`src/predictors.py`). That is why `evaluate.py` and `app.py` can treat them the same way, and why the app runs exactly the models that were evaluated.

## 1. Normalization (`src/normalize.py`), area B

Players hide bad words on purpose. Regular expressions undo the common tricks before any model sees the text:

| Trick | Regex step | Example |
|---|---|---|
| Accents, full-width letters, Turkish letters | Unicode NFKC/NFKD + a map for ı ş ğ ç ö ü | `ıdıot → idiot`, `ＩＤＩＯＴ → idiot` |
| Hashtags | `#(\w+)`, then split CamelCase | `#StopTheHate → stop the hate` |
| URLs, emoji | replace with `url` / remove | |
| Dotted / spaced letters | `\b(?:[a-z][.\-_*]){2,}[a-z]\b` | `f.u.c.k → fuck`, `s h i t → shit` |
| Leetspeak (only inside words that also contain letters) | map 0→o 1→i 3→e 4→a 5→s 7→t @→a $→s !→i | `id10t → idiot`, but `2019` stays `2019` |
| Repeated letters | `(.)\1{2,} → \1\1` | `stuuuupid → stuupid` (edit distance 1 from `stupid`) |

**Why keep 2 letters and not 1?** Real words have double letters (`good`, `kill`). Reducing to 2 never breaks them, and what remains is 1 edit away from the right word.

## 2. Model 1: lexicon + minimum edit distance (`src/lexicon.py`), area B, rule baseline

- **Lexicon** = the public LDNOOBW list (obscene words, CC BY 4.0) + `lexicon/extra_en.txt` (common insults that list lacks, such as idiot, stupid, loser).
- For each token: exact match → offensive. Otherwise compute the **minimum edit distance** to lexicon words of similar length with dynamic programming (a table `D[i][j]` = the cost to turn the first i letters into the first j letters). We use the **Damerau** version: swapping two neighbouring letters (`idoit`) costs 1, not 2.
- `*` in a message matches any letter for free: `f*ck` = distance 0 to `fuck`.
- **Only unknown words are fuzzy-matched.** Words seen ≥ 3 times in the training data are treated as correctly spelled and must match exactly. Without this, `rocks` would be 1 edit from an obscene word. This is the same idea as *non-word* spelling correction in the course.
- Short words (< 5 letters) need an exact match (`duck` ≠ `fuck`). Allowed distance: 1 for 5-7 letters, 2 for 8+.
- Confidence = similarity of the best match = 1 − distance / word length.
- **Weakness:** it cannot see context. "You are a disgrace" contains no bad word, while "that kill was f*cking amazing" is not an insult.

## 3. Model 2: TF-IDF + logistic regression (`src/tfidf_model.py`), area B, classical baseline

- **TF-IDF**: each message → a sparse vector with one dimension per word or word pair (1-2-grams seen in ≥ 2 training messages). Weight = (log-scaled) term frequency × inverse document frequency, so words that are common everywhere (`the`) get low weight and rare, telling words get high weight.
- **Logistic regression** learns one weight per feature, and P(offensive) = sigmoid(w·x + b).
- `class_weight="balanced"` makes mistakes on the rare class (offensive, about 1/3) cost more.
- **C** (inverse regularization strength) is picked on the **validation** set, never on the test set.

## 4. Model 3: our word2vec + feed-forward network (`src/w2v_ffn.py`), areas B + A

1. **word2vec (skip-gram)** is trained with gensim **only on the training messages**: each word learns a 100-number vector by predicting its neighbouring words (window 5, negative sampling). Words used in similar contexts get similar vectors. The script prints nearest neighbours as a check.
2. A message → **mean** of its word vectors (the general meaning) and **max** of each dimension (one strong word stands out) = 200 features. They are standardized with the training mean and std.
3. **Feed-forward network**: Linear(200→128) → ReLU → Dropout(0.3) → Linear(128→2).
4. Training: **SGD** with momentum (lr 0.01, mini-batches of 64). The loss is class-weighted cross-entropy, `loss.backward()` = **backpropagation** (chain rule computes the gradient of every weight), and `optimizer.step()` = w ← w − lr·gradient. After every epoch we measure validation macro-F1 and keep the best epoch (early stopping).
- **Weakness:** averaging loses word order. The vectors are learned from about 12k tweets only, so they are noisy.

## 5. Model 4: BERT fine-tuned (`src/bert_model.py`), area C

- `bert-base-uncased`: 12 Transformer encoder layers with **multi-head self-attention** (every token looks at every other token, so the meaning of a word depends on its context). It was pretrained on large English text with masked-language modelling.
- **WordPiece** subword tokenizer: a word that is not in the vocabulary is split into known pieces (pieces after the first start with `##`), so there are no out-of-vocabulary words. `src/bert_model.py` prints a real example at the start of training. Use it on your slide.
- A classification head (a linear layer) on the **[CLS]** token's final vector gives 2 logits.
- **Fine-tuning** = train all weights a little: AdamW, lr 2e-5, linear warm-up for 10 % of the steps then linear decay, 3 epochs, batch 32, max 64 tokens (tweets are short), class-weighted loss, gradient clipping at 1.0. Keep the epoch with the best validation macro-F1.

## 6. Model 5: quantization (`src/quantize.py`), area D

- FP32 weights take 4 bytes each, INT8 takes 1 byte. **Quantization**: f ≈ S · (q − Z), with scale S and zero point Z chosen from the range of the weights.
- **Dynamic** quantization: weights are converted to INT8 once, and activations are converted on the fly for each input. It needs no calibration data and no retraining, which is why the course recommends it for Transformers (they are limited by their huge weight matrices).
- Main path: export to **ONNX** (a standard graph format), then `onnxruntime.quantization.quantize_dynamic(..., QuantType.QInt8)`. ONNX Runtime also quantizes the word-embedding matrix and fuses operations, so it is usually smaller and faster than PyTorch's version, which only quantizes `nn.Linear` layers. Both are in the results table.
- Why faster on a CPU: INT8 matrix multiplication moves 4× less memory and uses fast integer instructions (AVX2/VNNI on x86, NEON on ARM).
- The script checks that INT8 and FP32 give the same label on 200 validation messages.

## 7. Evaluation (`src/evaluate.py`)

- **Same test set for all models**: the official OLID test split, never used for training or tuning.
- **Macro-F1** = the average of the F1 of "clean" and the F1 of "offensive". With imbalanced classes, a model that always says "clean" gets high accuracy but a bad macro-F1.
- **Size** = size of the model's files on disk (like `compute_size` in the course).
- **Latency** = one message at a time, CPU only, 10 warm-up runs, then 200 timed runs with `time.perf_counter()` over real test messages. We report mean ± std and the 95th percentile. It includes normalization + tokenization + the model (end-to-end).
- The hardware is printed under the table. Absolute numbers change with hardware, but the ranking usually does not.

## 8. The app (`app.py`)

- Gradio Blocks: chat panel (HTML), model dropdown, latency log table, per-model average latency, and an "all models on this message" table.
- Offensive messages are **masked** ("Message hidden by moderation"). A moderator can open the original.
- Latency in the app is measured around `predict_one` (normalization + model). At start-up every model is loaded and warmed up (3 runs), so loading is never counted and the first message is not a slow "cold start".
- If `models/` is missing, the app downloads the weights from the Hugging Face Hub repo in `config.MODEL_REPO`.

---

## Questions you may be asked, with short answers

**Why macro-F1 and not accuracy?** About 1/3 of the messages are offensive. Always predicting "clean" gives ≈ 70 % accuracy but finds nothing. Macro-F1 weights both classes equally.

**Why a validation set?** To choose C, the best epoch and the learning rate without looking at the test set. Otherwise the test score would be optimistic.

**Why is the INT8 model almost as accurate?** Transformer weights lie in a small range, so 256 levels represent them well. The rounding errors are small and mostly cancel out.

**Is ONNX Runtime INT8 the same as PyTorch INT8?** Both are dynamic INT8. ONNX Runtime also quantizes the embeddings and optimizes the graph (operator fusion), so it is usually smaller and faster. Compare the two rows in your table.

**Why does the lexicon have false alarms?** It has no context: a bad word inside a quote, or used as an intensifier in praise, still matches. And OLID labels some messages offensive that contain no bad word at all.

**Why did you use English and not Turkish?** The homework allows "Turkish (or English)". The normalization step already handles Turkish letters, and switching to OffensEval-TR + BERTurk only changes the dataset loader, `BERT_MODEL` and the lexicon (`config.py`).

**What would make it faster still?** Knowledge distillation into a smaller student (e.g. DistilBERT), shorter max length, ONNX graph optimizations, or batching messages on a busy server.

**What is the latency budget and did you meet it?** 50 ms per message on CPU. Answer from **your** table: which models are within budget and on which hardware.
