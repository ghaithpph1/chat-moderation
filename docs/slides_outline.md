# Presentation outline (PDF, 11 slides) + 2-minute video script

**Every number on the slides must come from YOUR run.** Copy it from `results/results.md`, `results/*.csv` or the notebook output. Never type a number from memory, from an AI answer or from a paper.
Placeholders look like **[from results.md]**.

---

## 1 · Title
- Real-Time Chat Moderation: classifying game-chat messages as clean / offensive on a CPU
- Your name · student ID · SWE015 · **Use Case 4 – Real-Time Chat Moderation**
- Repository link (GitHub) · Video link (YouTube unlisted / Drive) · *check both in a private window*

## INTRODUCTION

## 2 · Problem, users, and what success means
- Gaming platform: thousands of chat messages per minute. Offensive messages must be flagged **before** other players see them, on a **CPU-only** server.
- Users: players (protected) and moderators (see what was hidden and why).
- Success =
  1. **macro-F1** on a held-out test set (the classes are imbalanced, so accuracy would be misleading), and
  2. **≤ 50 ms per message on CPU** (our latency budget), reached with a **quantized** Transformer running in the app.

## DEVELOPMENT

## 3 · Data
- OLID / OffensEval-2019 English tweets, as packaged in TweetEval (Zampieri et al., 2019; Barbieri et al., 2020).
- Sizes and offensive share for train / validation / test: **[from the `python -m src.data` output]**
- Official test split = the ONE held-out test set for all models. Validation is used only for choosing hyper-parameters and the best epoch.
- Preprocessing: show 3 before → after examples from `python -m src.normalize`, e.g. `id10t → idiot`, `f.u.c.k → fuck`, `sooooo → soo`.

## 4 · Architecture diagram (R6)
- `docs/architecture.png`. Say: input → normalization → selected model (name + precision) → post-processing → UI, runs on CPU. The weights come from the Hugging Face Hub.

## 5 · Technique map (R2)
- Table from `docs/technique_map.md` (A, B, C, D → component → file).

## 6 · The four models and how each was trained
| | Model | Training |
|---|---|---|
| 1 | Lexicon + Damerau-Levenshtein | no training: public bad-word list + hand-added insults; fuzzy match only for unknown words |
| 2 | TF-IDF (1-2-grams) + logistic regression | balanced class weights, C chosen on validation **[best C from output]** |
| 3 | own word2vec (skip-gram, 100-d) → mean+max → FFN 128 → 2 | SGD (lr 0.01, momentum 0.9), dropout 0.3, class-weighted loss, best epoch on validation |
| 4 | bert-base-uncased fine-tuned | AdamW lr 2e-5, 3 epochs, batch 32, max 64 tokens, warm-up 10 % |
- Optional figure: `results/training_curves.png`

## 7 · Quantization (area D)
- Dynamic INT8: weights stored as 8-bit integers, activations quantized on the fly. No retraining.
- Pipeline: PyTorch FP32 → ONNX export → ONNX Runtime `quantize_dynamic` (also quantizes the embeddings).
- Extra row: PyTorch `quantize_dynamic` (Linear layers only), the same comparison as in the course lecture.
- FP32 vs INT8 agreement on 200 validation messages: **[from `python -m src.quantize` output]**

## 8 · Results table (R4)
- Paste `results/results.md`: model · macro-F1 · size (MB) · latency mean ± std (ms) · within budget.
- Hardware line under the table (it is printed under the table, and also in `results/hardware.txt`).

## CONCLUSION

## 9 · What the comparison showed
- Where each model **wins** and **loses**, for example: the lexicon is the fastest but misses insults that use no bad word. TF-IDF is strong and tiny. BERT has the best F1 but FP32 is slow and big. INT8 keeps (almost) the same F1 with a fraction of the size and latency.
- Write only what YOUR table shows. If BERT does **not** beat TF-IDF, say so and explain why (small dataset, tweets vs game chat, ...). That is a valid result.

## 10 · Limitations, failure example, next steps
- Failure examples: pick 1-2 rows from `results/errors_bert_int8_onnx.csv` (most confident mistakes are at the top). Show the text, the true label and the prediction, and explain why it fails (sarcasm, context, quoting an insult, a political word...).
- Limitations: tweets ≠ game chat (domain shift); English only; the lexicon has false alarms; latency depends on hardware.
- Next: Turkish data (OffensEval-TR 2020 + BERTurk), knowledge distillation into a smaller student, collect real game-chat data.

## 11 · AI usage (R8)
- Which tools, for which parts. Example: "Claude (Anthropic) was used to draft the code structure, the README and the slide outline. I ran all training and evaluation myself on Google Colab; every number comes from my runs. I reviewed and can explain every file." Edit it to be accurate.

---

# 2-minute video script (only the first 2:00 are watched)

| Time | Show | Say (example) |
|---|---|---|
| 0:00-0:15 | Title slide | "Game chats need offensive messages flagged in real time on a CPU server. I built a moderation app with four models of increasing power." |
| 0:15-1:15 | **The live app** (`python app.py`) | Type **new** messages: (1) a clean one: "gg wp, nice shot". (2) An obvious insult with leetspeak: "you are such an id10t", which is masked. (3) **A difficult case**: something like "you absolutely killed it" (violent word, but clean) or a sarcastic or quoted insult. Show the confidence, the latency log against 50 ms, switch the model to the lexicon and BERT FP32 to compare, and show the comparison table. |
| 1:15-1:45 | Architecture slide, then results table | "Messages are normalized with regex, then classified. The app runs BERT quantized to INT8 with ONNX Runtime. In my results it is [X]× smaller and [Y]× faster than FP32, with macro-F1 [Z]." (Your numbers.) |
| 1:45-2:00 | Conclusion slide | One sentence on what won, and one on the main limitation. |

Record with OBS / Xbox Game Bar / QuickTime, with voice narration. Upload as **unlisted** to YouTube (or Drive, "anyone with the link"). Open the link in a private window to check it.
