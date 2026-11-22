"""
Offline smoke test: runs the WHOLE pipeline + the app on a tiny synthetic
dataset with a tiny randomly initialized BERT, in a temporary folder.

It only checks that the code runs end to end. The numbers it prints are
meaningless (synthetic data) and must NOT be used in the presentation.

Run:  python tests/smoke_test.py
"""
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSULTS = ["idiot", "stupid", "loser", "moron", "dumb", "id10t", "stuuupid", "l0ser", "ıdiot", "f*ck"]
CLEAN = ["good game", "nice shot", "well played", "lets go team", "see you next round",
         "that was close", "great match everyone", "push mid now", "need healing", "thanks for the carry"]


def synthetic_split(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.33:
            rows.append((f"{rng.choice(CLEAN)} you {rng.choice(INSULTS)}", 1))
        else:
            rows.append((f"{rng.choice(CLEAN)} {rng.choice(CLEAN)}", 0))
    return pd.DataFrame(rows, columns=["text", "label"])


def make_tiny_bert(folder: Path, texts):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from src.normalize import normalize, tokenize

    words = sorted({w for t in texts for w in tokenize(normalize(t))} | {"url", "warm", "up", "message"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz") + \
            ["##" + c for c in "abcdefghijklmnopqrstuvwxyz"] + words
    folder.mkdir(parents=True)
    (folder / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)))
    tok = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128, num_labels=2)
    BertForSequenceClassification(cfg).save_pretrained(folder)
    tok.save_pretrained(folder)


def run(step, env):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", f"src.{step}"], cwd=ROOT, env=env, capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"  {step:<12} {status}  ({time.time() - t0:.1f}s)")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)
    return r.stdout


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cm_smoke_"))
    env = dict(os.environ, CM_DATA_DIR=str(tmp / "data"), CM_MODELS_DIR=str(tmp / "models"),
               CM_RESULTS_DIR=str(tmp / "results"), CM_BERT_MODEL=str(tmp / "tiny-bert"),
               CM_NO_DOWNLOAD="1", CM_BERT_EPOCHS="2", CM_W2V_EPOCHS="5", CM_FFN_EPOCHS="5",
               CM_LATENCY_RUNS="100", CM_MODEL_REPO="YOUR_HF_USERNAME/none",
               CM_ONNX_DYNAMO=os.environ.get("CM_ONNX_DYNAMO", "0"))
    print(f"Smoke test in {tmp}")

    (tmp / "data").mkdir()
    splits = {"train": synthetic_split(600, 1), "validation": synthetic_split(120, 2), "test": synthetic_split(150, 3)}
    for name, df in splits.items():
        df.to_csv(tmp / "data" / f"{name}.csv", index=False)
    make_tiny_bert(tmp / "tiny-bert", splits["train"]["text"])

    for step in ["lexicon", "tfidf_model", "w2v_ffn", "bert_model", "quantize", "evaluate"]:
        out = run(step, env)
    print(out[-2500:])

    results = pd.read_csv(tmp / "results" / "results.csv")
    assert len(results) >= 5, results
    assert {"macro_f1", "size_mb", "latency_ms"} <= set(results.columns)

    # --- the app: call the handler for every model, then start the server and fetch the page
    app_check = r'''
import os, time, urllib.request
import app
from src import predictors
chat, log = [], []
for key, name in predictors.available():
    for text in app.EXAMPLES[:3]:
        html_, _, chat, log, log_df, stats, cmp_df = app.on_send(text, name, True, chat, log)
assert len(log) == 3 * len(predictors.available()), len(log)
assert "Message hidden" in html_ or "clean" in html_
print("app handler ok:", len(log), "messages,", len(cmp_df), "models compared")
demo = app.build_demo()
demo.launch(server_port=7861, prevent_thread_lock=True)
time.sleep(2)
code = urllib.request.urlopen("http://127.0.0.1:7861/").status
print("app server http status:", code)
demo.close()
assert code == 200
'''
    r = subprocess.run([sys.executable, "-c", app_check], cwd=ROOT, env=env, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit("app check FAILED")
    print("\nSMOKE TEST PASSED (synthetic data - these numbers are not real results)")


if __name__ == "__main__":
    main()
