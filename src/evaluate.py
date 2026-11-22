"""
Evaluate every model of the ladder on the SAME held-out test set and build the
results table (task metric, size, latency) - requirement R4.

Benchmark method (same idea as the PerformanceBenchmark class in the course):
  * metric  : macro-F1 on the full test set (+ accuracy, precision/recall of "offensive")
  * size    : size of the model's files on disk, in MB
  * latency : one message at a time on the CPU, 10 warm-up runs, then
              LATENCY_RUNS timed runs (default 200) over real test messages;
              we report mean +- std and the 95th percentile, in milliseconds.

Outputs (in results/):
  results.md / results.csv     the table for your slides
  hardware.txt                 the machine the latency was measured on
  classification_reports.txt   per-class precision / recall / F1 for each model
  predictions_test.csv         every model's prediction for every test message
  errors_<model>.csv           misclassified test messages (pick failure examples here)

Run:  python -m src.evaluate            (all models that exist)
      python -m src.evaluate --runs 300
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import config
from src import predictors
from src.data import load_split
from src.utils import classification_metrics, hardware_info


def measure_latency(pred, texts, runs, warmup):
    for i in range(warmup):
        pred.predict_proba([texts[i % len(texts)]])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        pred.predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {"latency_ms": times.mean(), "latency_std": times.std(), "latency_p95": np.percentile(times, 95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=config.LATENCY_RUNS)
    parser.add_argument("--models", default="", help="comma-separated keys (default: all available)")
    args = parser.parse_args()

    test = load_split("test")
    texts, y = test["text"].tolist(), test["label"].to_numpy()
    keys = [k for k, _ in predictors.available()]
    if args.models:
        keys = [k for k in keys if k in args.models.split(",")]
    print(f"Test set: {len(test)} messages ({int(y.sum())} offensive). Models: {', '.join(keys)}\n")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hw = hardware_info()
    rows, reports = [], []
    all_preds = test[["text", "label"]].copy()

    for key in keys:
        print(f"== {key}")
        pred = predictors.load(key)
        proba = pred.predict_proba(texts)
        y_hat = proba.argmax(axis=1)
        metrics = classification_metrics(y, y_hat)
        latency = measure_latency(pred, texts, args.runs, config.WARMUP_RUNS)
        row = {"key": key, "model": pred.name, "area": pred.area, "precision": pred.precision,
               **metrics, "size_mb": pred.size_mb(), **latency}
        row["within_budget"] = row["latency_ms"] <= config.LATENCY_BUDGET_MS
        rows.append(row)
        print(f"   macro-F1 {row['macro_f1']:.4f} | size {row['size_mb']:.1f} MB | "
              f"latency {row['latency_ms']:.2f} +- {row['latency_std']:.2f} ms")

        reports.append(f"=== {pred.name} ===\n" + classification_report(
            y, y_hat, target_names=config.LABELS, digits=4, zero_division=0))
        all_preds[f"{key}_pred"] = [config.LABELS[i] for i in y_hat]
        all_preds[f"{key}_p_off"] = proba[:, 1].round(4)
        errors = test.assign(predicted=[config.LABELS[i] for i in y_hat], p_offensive=proba[:, 1])
        errors = errors[y_hat != y].copy()
        errors["true"] = [config.LABELS[i] for i in errors["label"]]
        errors["error_type"] = np.where(errors["label"] == 1, "missed offensive (FN)", "false alarm (FP)")
        # most confident mistakes first: the most interesting failure examples
        errors["confidence"] = np.where(errors["predicted"] == "offensive", errors["p_offensive"],
                                        1 - errors["p_offensive"])
        errors.sort_values("confidence", ascending=False)[
            ["text", "true", "predicted", "confidence", "error_type"]
        ].to_csv(config.RESULTS_DIR / f"errors_{key}.csv", index=False)
        del pred

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    all_preds.to_csv(config.RESULTS_DIR / "predictions_test.csv", index=False)
    (config.RESULTS_DIR / "classification_reports.txt").write_text("\n".join(reports))
    (config.RESULTS_DIR / "hardware.txt").write_text(json.dumps(hw, indent=2))

    hw_line = (f"Latency: batch size 1, CPU only, {config.WARMUP_RUNS} warm-up + {args.runs} timed runs "
               f"on {hw['cpu']} ({hw['logical_cores']} logical cores), "
               f"PyTorch {hw.get('torch', '-')}, ONNX Runtime {hw.get('onnxruntime', '-')}. "
               f"Latency budget: {config.LATENCY_BUDGET_MS:.0f} ms per message.")
    lines = [
        "| Model | Area | Precision | Macro-F1 | Accuracy | Size (MB) | Latency (ms) mean ± std | p95 (ms) | ≤ budget |",
        "|---|:-:|:-:|:-:|:-:|--:|--:|--:|:-:|",
    ]
    for r in rows:
        size = f"{r['size_mb']:.1f}" if r["size_mb"] >= 10 else f"{r['size_mb']:.2f}"
        lines.append(
            f"| {r['model']} | {r['area']} | {r['precision']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
            f"{size} | {r['latency_ms']:.2f} ± {r['latency_std']:.2f} | {r['latency_p95']:.2f} | "
            f"{'✅' if r['within_budget'] else '❌'} |"
        )
    table = "\n".join(lines) + f"\n\nTest set: {len(test)} messages (OLID test split). {hw_line}\n"
    (config.RESULTS_DIR / "results.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Saved everything to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
