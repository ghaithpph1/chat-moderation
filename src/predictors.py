"""
One common interface for every model, used by BOTH the evaluation script and
the app (so the app runs exactly the models that were evaluated).

    predictor.predict_proba(["raw message", ...]) -> array (n, 2) = [P(clean), P(offensive)]
    predictor.predict_one("raw message")          -> {"label", "confidence", "p_offensive"}
    predictor.size_mb()                           -> size of its files on disk

Every predictor takes RAW text and normalizes it itself, so the measured
latency is end-to-end (normalization + model).
Everything runs on the CPU, like the production server in the use case.
"""
import os
import warnings

import numpy as np

import config
from src.normalize import normalize
from src.utils import size_mb, softmax

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Predictor:
    key = ""      # short id
    name = ""     # display name
    area = ""     # technique area (A/B/C/D)
    precision = ""

    def predict_proba(self, texts) -> np.ndarray:
        raise NotImplementedError

    def files(self) -> list:
        raise NotImplementedError

    def size_mb(self) -> float:
        return size_mb(*self.files())

    def predict_one(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        label = int(p.argmax())
        return {"label": config.LABELS[label], "confidence": float(p[label]), "p_offensive": float(p[1])}


class LexiconPredictor(Predictor):
    key, name, area, precision = "lexicon", "Lexicon + edit distance (rule)", "B", "rules"

    def __init__(self):
        from src.lexicon import LexiconMatcher

        self.matcher = LexiconMatcher.load()

    def predict_proba(self, texts):
        out = np.zeros((len(texts), 2))
        for i, t in enumerate(texts):
            hits = self.matcher.find(t)
            p_off = max((sim for _, _, sim in hits), default=0.0)  # similarity of the best match
            out[i] = [1.0 - p_off, p_off]
        return out

    def files(self):
        return [config.LEXICON_PATH, config.KNOWN_WORDS_PATH]


class TfidfPredictor(Predictor):
    key, name, area, precision = "tfidf", "TF-IDF + logistic regression", "B", "FP64"

    def __init__(self):
        import joblib

        self.model = joblib.load(config.TFIDF_PATH)

    def predict_proba(self, texts):
        return self.model.predict_proba(list(texts))  # the pipeline normalizes internally

    def files(self):
        return [config.TFIDF_PATH]


class W2VFFNPredictor(Predictor):
    key, name, area, precision = "w2v_ffn", "Word2Vec + feed-forward NN", "A", "FP32"

    def __init__(self):
        import torch

        from src.w2v_ffn import FeedForward, WordVectors

        self.torch = torch
        self.wv = WordVectors.load()
        ckpt = torch.load(config.FFN_PATH, map_location="cpu")
        self.model = FeedForward(ckpt["in_dim"], ckpt["hidden"], ckpt["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mu, self.sd = ckpt["mu"].numpy(), ckpt["sd"].numpy()

    def predict_proba(self, texts):
        x = (self.wv.featurize(texts) - self.mu) / self.sd
        with self.torch.inference_mode():
            return self.model(self.torch.tensor(x, dtype=self.torch.float32)).softmax(-1).numpy()

    def files(self):
        return [config.W2V_VOCAB_PATH, config.W2V_VECTORS_PATH, config.FFN_PATH]


class BertTorchPredictor(Predictor):
    """BERT in PyTorch: FP32, or INT8 via torch.ao.quantization (optional extra row)."""

    def __init__(self, int8: bool = False):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.int8 = torch, int8
        if int8:
            from torch.ao.quantization import quantize_dynamic

            self.key, self.name, self.area, self.precision = "bert_int8_torch", "BERT INT8 (PyTorch dynamic)", "D", "INT8"
            folder = config.BERT_INT8_TORCH_DIR
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skeleton = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(folder))
                self.model = quantize_dynamic(skeleton, {torch.nn.Linear}, dtype=torch.qint8)
                # our own file, written by src/quantize.py -> safe to unpickle
                self.model.load_state_dict(torch.load(folder / "model_int8.pt", map_location="cpu",
                                                      weights_only=False))
        else:
            self.key, self.name, self.area, self.precision = "bert_fp32", "BERT-base fine-tuned (FP32)", "C", "FP32"
            folder = config.BERT_FP32_DIR
            self.model = AutoModelForSequenceClassification.from_pretrained(folder)
        self.folder = folder
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model.to("cpu").eval()

    def predict_proba(self, texts, batch_size=32):
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                batch = [normalize(t) for t in texts[s: s + batch_size]]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=config.BERT_MAX_LEN, return_tensors="pt")
                out.append(self.model(**enc).logits.softmax(-1).numpy())
        return np.concatenate(out)

    def files(self):
        if self.int8:
            return [self.folder / "model_int8.pt"]
        return list(self.folder.glob("*.safetensors")) + list(self.folder.glob("*.bin"))


class BertOnnxPredictor(Predictor):
    """BERT INT8 with ONNX Runtime - the model the app runs by default."""
    key, name, area, precision = "bert_int8_onnx", "BERT INT8 (ONNX Runtime)", "D", "INT8"

    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.path = config.BERT_INT8_ONNX_DIR / "model_int8.onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_INT8_ONNX_DIR)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def predict_proba(self, texts, batch_size=32):
        out = []
        for s in range(0, len(texts), batch_size):
            batch = [normalize(t) for t in texts[s: s + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=config.BERT_MAX_LEN, return_tensors="np")
            feeds = {n: enc[n].astype(np.int64) for n in self.input_names}
            out.append(softmax(self.session.run(None, feeds)[0]))
        return np.concatenate(out)

    def files(self):
        return [self.path]


# ----------------------------------------------------------------------------
# Registry: (key, display name, constructor, file that must exist)
# Order = the model ladder, as it appears in the results table and the app.
# ----------------------------------------------------------------------------
REGISTRY = [
    ("lexicon", LexiconPredictor.name, LexiconPredictor, config.LEXICON_PATH),
    ("tfidf", TfidfPredictor.name, TfidfPredictor, config.TFIDF_PATH),
    ("w2v_ffn", W2VFFNPredictor.name, W2VFFNPredictor, config.FFN_PATH),
    ("bert_fp32", "BERT-base fine-tuned (FP32)", lambda: BertTorchPredictor(int8=False),
     config.BERT_FP32_DIR / "config.json"),
    ("bert_int8_onnx", BertOnnxPredictor.name, BertOnnxPredictor,
     config.BERT_INT8_ONNX_DIR / "model_int8.onnx"),
    ("bert_int8_torch", "BERT INT8 (PyTorch dynamic)", lambda: BertTorchPredictor(int8=True),
     config.BERT_INT8_TORCH_DIR / "model_int8.pt"),
]
DEFAULT_KEY = "bert_int8_onnx"
SHORT_NAMES = {"lexicon": "Lexicon", "tfidf": "TF-IDF", "w2v_ffn": "W2V+FFN", "bert_fp32": "BERT FP32",
               "bert_int8_onnx": "BERT INT8 ORT", "bert_int8_torch": "BERT INT8 PT"}


def available() -> list:
    """(key, name) of every model whose files exist on disk."""
    return [(key, name) for key, name, _, path in REGISTRY if path.exists()]


def load(key: str) -> Predictor:
    for k, _, ctor, path in REGISTRY:
        if k == key:
            if not path.exists():
                raise FileNotFoundError(f"{key}: {path} not found - train/download the models first")
            return ctor()
    raise KeyError(key)
