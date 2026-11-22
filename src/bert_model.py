"""
Model 4 (Transformer, area C): fine-tune BERT for offensive / clean classification.

We take pretrained bert-base-uncased (WordPiece tokenizer, 12 layers of
self-attention) and put a classification head on the [CLS] token. Then we
fine-tune ALL weights for a few epochs on the training tweets. After each
epoch we check macro-F1 on the validation set and keep the best epoch.

Run (GPU recommended - Colab T4 takes a few minutes):  python -m src.bert_model
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from src.data import load_split
from src.normalize import normalize
from src.utils import class_weights, classification_metrics, set_seed


def batches(texts, labels, batch_size, shuffle, seed=0):
    order = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start: start + batch_size]
        yield [texts[i] for i in idx], torch.tensor([labels[i] for i in idx])


@torch.inference_mode()
def predict(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    preds = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start: start + batch_size], padding=True, truncation=True,
                        max_length=config.BERT_MAX_LEN, return_tensors="pt").to(device)
        preds.append(model(**enc).logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                 "  -- no GPU: this will be slow, use Colab with a T4 GPU"))

    train, val = load_split("train"), load_split("validation")
    train_texts = [normalize(t) for t in train["text"]]
    val_texts = [normalize(t) for t in val["text"]]
    train_labels = train["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL, num_labels=2,
        id2label={i: l for i, l in enumerate(config.LABELS)},
        label2id={l: i for i, l in enumerate(config.LABELS)},
    ).to(device)
    print(f"Model: {config.BERT_MODEL}  ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")
    example = tokenizer.tokenize(normalize("This game is sooo stuupid lol"))
    print(f"WordPiece example: {example}")

    steps_per_epoch = (len(train_texts) + config.BERT_BATCH - 1) // config.BERT_BATCH
    total_steps = steps_per_epoch * config.BERT_EPOCHS
    warmup_steps = int(config.BERT_WARMUP_RATIO * total_steps)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": config.BERT_WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=config.BERT_LR)

    def lr_lambda(step):  # linear warm-up, then linear decay to 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train_labels), dtype=torch.float32).to(device))

    best_f1, history = -1.0, []
    for epoch in range(1, config.BERT_EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (texts, labels) in enumerate(
                batches(train_texts, train_labels, config.BERT_BATCH, shuffle=True, seed=config.SEED + epoch), 1):
            enc = tokenizer(texts, padding=True, truncation=True, max_length=config.BERT_MAX_LEN,
                            return_tensors="pt").to(device)
            loss = loss_fn(model(**enc).logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == steps_per_epoch:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch}  loss {running / step:.4f}")

        val_pred = predict(model, tokenizer, val_texts, device)
        f1 = classification_metrics(val["label"], val_pred)["macro_f1"]
        print(f"epoch {epoch}: val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_macro_f1": f1,
                        "seconds": round(time.time() - t0)})
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(config.RESULTS_DIR / "train_history_bert.csv", index=False)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(config.BERT_FP32_DIR)
            tokenizer.save_pretrained(config.BERT_FP32_DIR)
            print(f"  -> best so far, saved to {config.BERT_FP32_DIR}")

    print(f"\nBest val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
