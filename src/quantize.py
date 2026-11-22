"""
Model 5 (efficient model, area D): quantize the fine-tuned BERT to INT8.

Dynamic quantization (the method the course recommends for Transformers):
the weights are stored as 8-bit integers (4x smaller than 32-bit floats), and
activations are quantized on the fly at inference time. No re-training needed.

Main path (used by the app):
    PyTorch FP32 model -> export to ONNX -> ONNX Runtime quantize_dynamic (INT8)
    ONNX Runtime also quantizes the embedding matrix, so the file is smaller,
    and it runs the same on Windows / macOS / Linux CPUs.
Extra row for the results table (if your PyTorch version still supports it):
    torch.ao.quantization.quantize_dynamic on all nn.Linear layers.

Run (CPU is fine):  python -m src.quantize
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from src.data import load_split
from src.normalize import normalize
from src.utils import size_mb

ONNX_FP32 = config.BERT_ONNX_FP32_DIR / "model.onnx"
ONNX_INT8 = config.BERT_INT8_ONNX_DIR / "model_int8.onnx"
TORCH_INT8 = config.BERT_INT8_TORCH_DIR / "model_int8.pt"


class LogitsOnly(nn.Module):
    """Wrapper with a fixed argument order, so the ONNX inputs get the right names."""

    def __init__(self, model, use_token_type_ids):
        super().__init__()
        self.model, self.use_tt = model, use_token_type_ids

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_tt:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs).logits


def export_onnx(model, tokenizer):
    config.BERT_ONNX_FP32_DIR.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(["export example message"], return_tensors="pt")
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in sample else [])
    wrapper = LogitsOnly(model, "token_type_ids" in names).eval()
    args = tuple(sample[n] for n in names)

    try:  # classic TorchScript-based exporter: one self-contained .onnx file
        if os.environ.get("CM_ONNX_DYNAMO") == "1":  # (lets the smoke test try the other exporter)
            raise RuntimeError("dynamo exporter requested")
        torch.onnx.export(
            wrapper, args, str(ONNX_FP32), input_names=names, output_names=["logits"],
            dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
            opset_version=config.ONNX_OPSET, dynamo=False,
        )
        print("Exported with the TorchScript ONNX exporter")
    except Exception as e:  # newer PyTorch: use the torch.export-based exporter
        print(f"TorchScript exporter not available ({type(e).__name__}); using dynamo=True exporter")
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
        torch.onnx.export(
            wrapper, args, str(ONNX_FP32), input_names=names, output_names=["logits"],
            dynamic_shapes={n: {0: batch, 1: seq} for n in names}, dynamo=True,
        )
    return names


def quantize_onnx():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    config.BERT_INT8_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QInt8)


def quantize_torch(model):
    """Optional: PyTorch dynamic quantization of every nn.Linear layer."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        print("torch.ao.quantization is not available in this PyTorch version - skipping the extra row.")
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    config.BERT_INT8_TORCH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), TORCH_INT8)
    model.config.save_pretrained(config.BERT_INT8_TORCH_DIR)
    return True


def main():
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_FP32_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.BERT_FP32_DIR).to("cpu").eval()

    print("1) Export FP32 model to ONNX")
    export_onnx(model, tokenizer)
    print("2) ONNX Runtime dynamic quantization -> INT8")
    quantize_onnx()
    tokenizer.save_pretrained(config.BERT_INT8_ONNX_DIR)  # the app needs the tokenizer next to the model
    model.config.save_pretrained(config.BERT_INT8_ONNX_DIR)  # (keeps the label names)
    print("3) PyTorch dynamic quantization (optional extra)")
    has_torch_int8 = quantize_torch(model)
    if has_torch_int8:
        tokenizer.save_pretrained(config.BERT_INT8_TORCH_DIR)

    print("\nSize on disk:")
    print(f"  BERT FP32 (safetensors)  {size_mb(*config.BERT_FP32_DIR.glob('*.safetensors')):8.1f} MB")
    print(f"  BERT FP32 (ONNX)         {size_mb(config.BERT_ONNX_FP32_DIR):8.1f} MB")
    print(f"  BERT INT8 (ONNX Runtime) {size_mb(ONNX_INT8):8.1f} MB")
    if has_torch_int8:
        print(f"  BERT INT8 (PyTorch)      {size_mb(TORCH_INT8):8.1f} MB")

    # Sanity check: does the INT8 model still agree with the FP32 model?
    import onnxruntime as ort
    from src.utils import softmax

    texts = [normalize(t) for t in load_split("validation")["text"][:200]]
    session = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    agree = 0
    with torch.inference_mode():
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=config.BERT_MAX_LEN, return_tensors="np")
            p_int8 = softmax(session.run(None, {n: enc[n].astype(np.int64) for n in input_names})[0])
            p_fp32 = model(**tokenizer(t, truncation=True, max_length=config.BERT_MAX_LEN,
                                       return_tensors="pt")).logits.softmax(-1).numpy()
            agree += int(p_int8.argmax() == p_fp32.argmax())
    print(f"\nINT8 and FP32 predict the same label on {agree}/{len(texts)} validation messages")


if __name__ == "__main__":
    main()
