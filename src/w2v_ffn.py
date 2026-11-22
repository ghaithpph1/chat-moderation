"""
Model 3 (neural network, area A): our own word2vec + a feed-forward network.

1. Train word2vec (skip-gram, gensim) on the TRAINING messages only, so every
   word gets a 100-dimensional vector learned from this data.
2. Represent a message by the mean and the max of its word vectors (200 numbers).
3. Train a small feed-forward network on those features with plain SGD and
   backpropagation:   200 -> Linear -> ReLU -> Dropout -> Linear -> 2 classes

Run:  python -m src.w2v_ffn
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from src.data import load_split
from src.normalize import normalize, tokenize
from src.utils import class_weights, classification_metrics, set_seed


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits; softmax is applied inside the loss / at prediction time


class WordVectors:
    """Lookup table word -> vector, stored as plain JSON + NumPy (no gensim needed at inference)."""

    def __init__(self, vocab: dict, vectors: np.ndarray):
        self.vocab, self.vectors = vocab, vectors
        self.dim = vectors.shape[1]

    def featurize(self, texts) -> np.ndarray:
        feats = np.zeros((len(texts), 2 * self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            idx = [self.vocab[t] for t in tokenize(normalize(text)) if t in self.vocab]
            if idx:
                vecs = self.vectors[idx]
                feats[i, : self.dim] = vecs.mean(axis=0)
                feats[i, self.dim:] = vecs.max(axis=0)
        return feats

    def save(self):
        config.W2V_VOCAB_PATH.write_text(json.dumps(self.vocab), encoding="utf-8")
        np.save(config.W2V_VECTORS_PATH, self.vectors)

    @classmethod
    def load(cls):
        vocab = json.loads(config.W2V_VOCAB_PATH.read_text(encoding="utf-8"))
        return cls(vocab, np.load(config.W2V_VECTORS_PATH))


def train_word2vec(train_texts) -> WordVectors:
    from gensim.models import Word2Vec

    sentences = [tokenize(normalize(t)) for t in train_texts]
    w2v = Word2Vec(sentences=sentences, vector_size=config.W2V_DIM, window=config.W2V_WINDOW,
                   min_count=config.W2V_MIN_COUNT, sg=1, negative=5, epochs=config.W2V_EPOCHS,
                   seed=config.SEED, workers=1)
    print(f"word2vec: {len(w2v.wv)} words x {config.W2V_DIM} dims (skip-gram, {config.W2V_EPOCHS} epochs)")
    for probe in ["idiot", "stupid", "game", "love"]:
        if probe in w2v.wv:
            neighbours = ", ".join(w for w, _ in w2v.wv.most_similar(probe, topn=5))
            print(f"  nearest to {probe!r}: {neighbours}")
    vocab = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
    return WordVectors(vocab, w2v.wv.vectors.astype(np.float32))


def main():
    set_seed(config.SEED)
    train, val = load_split("train"), load_split("validation")

    wv = train_word2vec(train["text"])
    x_train, x_val = wv.featurize(train["text"]), wv.featurize(val["text"])
    # standardize features (mean 0, std 1), using TRAINING statistics only
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    x_train, x_val = (x_train - mu) / sd, (x_val - mu) / sd
    y_train = torch.tensor(train["label"].values)

    model = FeedForward(x_train.shape[1], config.FFN_HIDDEN, config.FFN_DROPOUT)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.FFN_LR,
                                momentum=config.FFN_MOMENTUM, weight_decay=config.FFN_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train["label"]), dtype=torch.float32))
    xt = torch.tensor(x_train)
    xv = torch.tensor(x_val)

    best_f1, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(1, config.FFN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), config.FFN_BATCH):
            idx = perm[start: start + config.FFN_BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), y_train[idx])  # forward pass
            loss.backward()                                # backpropagation
            optimizer.step()                               # SGD update
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(xv).argmax(dim=1).numpy()
        f1 = classification_metrics(val["label"], val_pred)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d}  train loss {total / len(xt):.4f}  val macro-F1 {f1:.4f}")

    wv.save()
    torch.save({"state_dict": best_state, "mu": torch.tensor(mu), "sd": torch.tensor(sd),
                "in_dim": x_train.shape[1],
                "hidden": config.FFN_HIDDEN, "dropout": config.FFN_DROPOUT}, config.FFN_PATH)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.RESULTS_DIR / "train_history_w2v_ffn.csv", index=False)
    print(f"\nBest epoch {best_epoch} (val macro-F1 {best_f1:.4f})")
    print(f"Saved to {config.W2V_VECTORS_PATH.name}, {config.W2V_VOCAB_PATH.name}, {config.FFN_PATH.name}")


if __name__ == "__main__":
    main()
