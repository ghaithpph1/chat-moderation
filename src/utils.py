"""Small helpers shared by all scripts: seeding, file sizes, metrics, hardware info."""
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Make runs repeatable (same seed -> same split of mini-batches, same init)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def size_mb(*paths) -> float:
    """Size on disk in MB of files and/or folders (folders are summed recursively)."""
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total / (1024 * 1024)


def classification_metrics(y_true, y_pred) -> dict:
    """Macro-F1 is the main metric because the classes are imbalanced."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_off": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_off": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_off": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def class_weights(labels) -> np.ndarray:
    """Weights n / (2 * count_c): the rarer class (offensive) gets a larger weight."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    return len(labels) / (2.0 * counts)


def cpu_name() -> str:
    """Human-readable CPU name (for the 'hardware' line under the results table)."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def hardware_info() -> dict:
    info = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for package in ["onnxruntime", "transformers", "sklearn", "gradio"]:
        try:
            info[package] = __import__(package).__version__
        except ImportError:
            pass
    return info


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
