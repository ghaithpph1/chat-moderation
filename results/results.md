| Model | Area | Precision | Macro-F1 | Accuracy | Size (MB) | Latency (ms) mean ± std | p95 (ms) | ≤ budget |
|---|:-:|:-:|:-:|:-:|--:|--:|--:|:-:|
| Lexicon + edit distance (rule) | B | rules | 0.677 | 0.774 | 0.04 | 0.18 ± 0.10 | 0.37 | ✅ |
| TF-IDF + logistic regression | B | FP64 | 0.728 | 0.771 | 0.94 | 0.98 ± 0.15 | 1.22 | ✅ |
| Word2Vec + feed-forward NN | A | FP32 | 0.650 | 0.690 | 3.48 | 0.57 ± 0.21 | 0.87 | ✅ |
| BERT-base fine-tuned (FP32) | C | FP32 | 0.808 | 0.844 | 417.7 | 136.15 ± 31.71 | 194.71 | ❌ |
| BERT INT8 (ONNX Runtime) | D | INT8 | 0.786 | 0.843 | 105.2 | 53.90 ± 24.33 | 97.27 | ❌ |
| BERT INT8 (PyTorch dynamic) | D | INT8 | 0.767 | 0.837 | 173.1 | 62.83 ± 25.09 | 105.35 | ❌ |

Test set: 860 messages (OLID test split). Latency: batch size 1, CPU only, 10 warm-up + 200 timed runs on Intel(R) Xeon(R) CPU @ 2.00GHz (2 logical cores), PyTorch 2.14.1+cu130, ONNX Runtime 1.32.0. Latency budget: 50 ms per message.
