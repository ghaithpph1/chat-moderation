"""
Central configuration for the Real-Time Chat Moderation project.

Every script imports its paths and settings from here, so there is exactly
one place to change a hyper-parameter. A few values can be overridden with
environment variables (used by the smoke test, and handy on Colab).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value is not None else default


# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
DATA_DIR = Path(_env("CM_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(_env("CM_MODELS_DIR", ROOT / "models"))
RESULTS_DIR = Path(_env("CM_RESULTS_DIR", ROOT / "results"))

SEED = 42
LABELS = ["clean", "offensive"]  # label 0 = clean (NOT offensive), 1 = offensive (OFF)

# ----------------------------------------------------------------------------
# Data: OLID / OffensEval-2019 English tweets, as packaged in TweetEval
# ----------------------------------------------------------------------------
HF_DATASET = ("cardiffnlp/tweet_eval", "offensive")
GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/"
    "datasets/offensive/{split}_{kind}.txt"
)
# Public bad-word list (LDNOOBW, CC BY 4.0) used by the rule baseline.
BADWORDS_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)
NO_DOWNLOAD = _env("CM_NO_DOWNLOAD", "0") == "1"  # smoke test: stay offline

# ----------------------------------------------------------------------------
# Model 1 - rule baseline: lexicon + minimum edit distance
# ----------------------------------------------------------------------------
EXTRA_LEXICON_PATH = ROOT / "lexicon" / "extra_en.txt"  # hand-added insults (in the repo)
LEXICON_FALLBACK_TOP_K = 150  # only used if the public list cannot be downloaded
FUZZY_MIN_LEN = 5             # words shorter than this must match exactly
KNOWN_WORD_MIN_COUNT = 3      # training-set words seen this often count as "correctly spelled"

# ----------------------------------------------------------------------------
# Model 2 - classical baseline: TF-IDF + logistic regression
# ----------------------------------------------------------------------------
TFIDF_NGRAMS = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# ----------------------------------------------------------------------------
# Model 3 - neural network: our own word2vec + feed-forward network (SGD)
# ----------------------------------------------------------------------------
W2V_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = _env("CM_W2V_EPOCHS", 20, int)
FFN_HIDDEN = 128
FFN_DROPOUT = 0.3
FFN_LR = 0.01
FFN_MOMENTUM = 0.9
FFN_WEIGHT_DECAY = 1e-4
FFN_EPOCHS = _env("CM_FFN_EPOCHS", 40, int)
FFN_BATCH = 64

# ----------------------------------------------------------------------------
# Model 4 - Transformer: BERT fine-tuned for classification
# ----------------------------------------------------------------------------
BERT_MODEL = _env("CM_BERT_MODEL", "google-bert/bert-base-uncased")
BERT_MAX_LEN = 64
BERT_EPOCHS = _env("CM_BERT_EPOCHS", 3, int)
BERT_BATCH = 32
BERT_LR = 2e-5
BERT_WARMUP_RATIO = 0.1
BERT_WEIGHT_DECAY = 0.01

# ----------------------------------------------------------------------------
# Model 5 - efficient model: quantized BERT + latency budget
# ----------------------------------------------------------------------------
LATENCY_BUDGET_MS = 50.0   # our target: <= 50 ms per message on a CPU
WARMUP_RUNS = 10           # untimed runs before measuring (like the course benchmark)
LATENCY_RUNS = _env("CM_LATENCY_RUNS", 200, int)  # timed runs (homework asks for >= 100)
ONNX_OPSET = 17

# ----------------------------------------------------------------------------
# Where every trained artifact is stored
# ----------------------------------------------------------------------------
LEXICON_PATH = MODELS_DIR / "lexicon.txt"
KNOWN_WORDS_PATH = MODELS_DIR / "known_words.txt"
TFIDF_PATH = MODELS_DIR / "tfidf_logreg.joblib"
W2V_VOCAB_PATH = MODELS_DIR / "w2v_vocab.json"
W2V_VECTORS_PATH = MODELS_DIR / "w2v_vectors.npy"
FFN_PATH = MODELS_DIR / "w2v_ffn.pt"
BERT_FP32_DIR = MODELS_DIR / "bert_fp32"
BERT_ONNX_FP32_DIR = MODELS_DIR / "bert_onnx_fp32"   # intermediate, not used by the app
BERT_INT8_ONNX_DIR = MODELS_DIR / "bert_int8_onnx"   # the model the app runs by default
BERT_INT8_TORCH_DIR = MODELS_DIR / "bert_int8_torch"  # optional extra row

# Hugging Face Hub repository that stores the trained weights (weights are NOT
# committed to Git). Change this to "<your-hf-username>/chat-moderation-models"
# after you upload them, or set the CM_MODEL_REPO environment variable.
MODEL_REPO = _env("CM_MODEL_REPO", "ghaithpph/chat.moderation.models")
