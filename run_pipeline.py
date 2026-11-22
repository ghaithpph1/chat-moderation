"""
Run the whole training + evaluation pipeline in order.

    python run_pipeline.py                  # everything
    python run_pipeline.py --from quantize  # resume from a step
    python run_pipeline.py --only evaluate  # a single step
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("data", "download OLID / TweetEval and save train/validation/test CSVs"),
    ("lexicon", "Model 1: build the bad-word lexicon (rule baseline)"),
    ("tfidf_model", "Model 2: TF-IDF + logistic regression"),
    ("w2v_ffn", "Model 3: our word2vec + feed-forward network (SGD)"),
    ("bert_model", "Model 4: fine-tune BERT (GPU recommended)"),
    ("quantize", "Model 5: export to ONNX + INT8 dynamic quantization"),
    ("evaluate", "evaluate all models on the test set -> results/results.md"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
    parser.add_argument("--only", choices=[s for s, _ in STEPS])
    args = parser.parse_args()

    names = [s for s, _ in STEPS]
    todo = [args.only] if args.only else names[names.index(args.start):] if args.start else names
    for step, desc in STEPS:
        if step not in todo:
            continue
        print(f"\n{'=' * 78}\n>>> {step}: {desc}\n{'=' * 78}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, "-m", f"src.{step}"])
        if result.returncode != 0:
            sys.exit(f"Step '{step}' failed (exit code {result.returncode}).")
        print(f"<<< {step} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
