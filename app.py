"""
Real-Time Chat Moderation - Gradio application.

Type chat messages: each one is labeled live (clean / offensive) with a
confidence score, offensive messages are masked, and the side panel logs the
response time of every message against the latency budget. You can switch
models at any time. The default is the quantized BERT (INT8, ONNX Runtime).

Run:  python app.py            (opens http://127.0.0.1:7860)
      python app.py --share    (public link, e.g. from Google Colab)
"""
import argparse
import html
import time

import gradio as gr
import pandas as pd

import config
from src import predictors

EXAMPLES = [
    "gg wp everyone, that last round was close!",
    "you are such an id10t, uninstall the game",
    "this lag is sooooo stuuupid",
    "nice shot, you absolutely killed it",
    "@USER shut up loser nobody asked you",
    "f*ck this team",
    "I will destroy you in the next match :)",
]
LOG_COLUMNS = ["#", "model", "label", "conf.", "ms", "budget"]
_loaded = {}  # key -> Predictor (loaded once, on first use)


def ensure_models():
    """Download the trained weights from the Hugging Face Hub if they are not on disk."""
    if predictors.available() and (config.BERT_INT8_ONNX_DIR / "model_int8.onnx").exists():
        return
    if config.MODEL_REPO.startswith("YOUR_HF_USERNAME"):
        print("No models found in models/ and MODEL_REPO is not set in config.py.\n"
              "Train them (see README) or set MODEL_REPO to your Hugging Face repo.")
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading trained models from https://huggingface.co/{config.MODEL_REPO} ...")
    snapshot_download(repo_id=config.MODEL_REPO, local_dir=str(config.MODELS_DIR))


def get(key):
    if key not in _loaded:
        _loaded[key] = predictors.load(key)
    return _loaded[key]


def name_to_key(name):
    return {n: k for k, n in predictors.available()}[name]


def render_chat(chat):
    if not chat:
        body = "<div style='opacity:.6;padding:24px;text-align:center'>No messages yet - type one below.</div>"
    else:
        items = []
        for m in reversed(chat):  # column-reverse: newest message at the bottom, scrolled into view
            offensive = m["label"] == "offensive"
            color = "#d64545" if offensive else "#2e9e5b"
            badge = (f"<span style='background:{color};color:white;border-radius:10px;padding:1px 8px;"
                     f"font-size:12px'>{m['label']} · {m['confidence']:.0%}</span>")
            meta = f"<span style='opacity:.6;font-size:12px'> {html.escape(m['model'])} · {m['ms']:.1f} ms</span>"
            if offensive:
                text = ("<i>🚫 Message hidden by moderation</i>"
                        "<details style='font-size:12px;opacity:.7'><summary>show original (moderator)</summary>"
                        f"{html.escape(m['text'])}</details>")
            else:
                text = html.escape(m["text"])
            items.append(
                f"<div style='margin:6px 0;padding:8px 12px;border-radius:12px;"
                f"border-left:4px solid {color};background:rgba(127,127,127,.08)'>"
                f"<b>Player:</b> {text}<div style='margin-top:4px'>{badge}{meta}</div></div>")
        body = "".join(items)
    return ("<div style='height:380px;overflow-y:auto;display:flex;flex-direction:column-reverse;"
            f"padding:6px;border:1px solid rgba(127,127,127,.3);border-radius:10px'><div>{body}</div></div>")


def stats_markdown(log):
    if not log:
        return f"**Latency budget:** {config.LATENCY_BUDGET_MS:.0f} ms per message (CPU)"
    df = pd.DataFrame(log, columns=LOG_COLUMNS)
    lines = [f"**Latency budget:** {config.LATENCY_BUDGET_MS:.0f} ms per message (CPU)", "",
             "| model | messages | mean latency (ms) |", "|---|--:|--:|"]
    for model, g in df.groupby("model", sort=False):
        lines.append(f"| {model} | {len(g)} | {g['ms'].mean():.1f} |")
    return "\n".join(lines)


def on_send(text, model_name, compare, chat, log):
    text = (text or "").strip()
    if not text:
        return render_chat(chat), "", chat, log, pd.DataFrame(log, columns=LOG_COLUMNS), stats_markdown(log), gr.update()

    key = name_to_key(model_name)
    pred = get(key)                           # loading is NOT counted in the latency
    t0 = time.perf_counter()
    out = pred.predict_one(text)              # normalization + model
    ms = (time.perf_counter() - t0) * 1000.0

    chat = chat + [{"text": text, "label": out["label"], "confidence": out["confidence"],
                    "model": pred.name, "ms": ms}]
    log = log + [[len(log) + 1, predictors.SHORT_NAMES.get(key, pred.name), out["label"],
                  round(out["confidence"], 3), round(ms, 2), "✅" if ms <= config.LATENCY_BUDGET_MS else "❌"]]

    comparison = gr.update()
    if compare:
        rows = []
        for key, name in predictors.available():
            p = get(key)
            t0 = time.perf_counter()
            o = p.predict_one(text)
            rows.append([name, o["label"], round(o["confidence"], 3), round((time.perf_counter() - t0) * 1000, 2)])
        comparison = pd.DataFrame(rows, columns=["model", "label", "confidence", "latency (ms)"])

    return (render_chat(chat), "", chat, log, pd.DataFrame(log[::-1], columns=LOG_COLUMNS),
            stats_markdown(log), comparison)


def on_clear():
    return render_chat([]), [], [], pd.DataFrame(columns=LOG_COLUMNS), stats_markdown([]), \
        pd.DataFrame(columns=["model", "label", "confidence", "latency (ms)"])


def build_demo():
    models = predictors.available()
    names = [n for _, n in models]
    default = dict(models).get(predictors.DEFAULT_KEY, names[0] if names else None)

    with gr.Blocks(title="Real-Time Chat Moderation") as demo:
        gr.Markdown(
            "# 🎮 Real-Time Chat Moderation\n"
            "Every message is classified **live** as *clean* or *offensive* on the CPU. "
            "Offensive messages are masked. The default model is **BERT quantized to INT8 (ONNX Runtime)**.")
        chat_state, log_state = gr.State([]), gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat_html = gr.HTML(render_chat([]))
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type a chat message and press Enter…", show_label=False,
                                     scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)
                compare = gr.Checkbox(value=True, label="Also run every model on this message (side by side)")
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[msg], label="Try these")
            with gr.Column(scale=2):
                model_dd = gr.Dropdown(choices=names, value=default, label="Moderation model")
                stats = gr.Markdown(stats_markdown([]))
                log_df = gr.Dataframe(value=pd.DataFrame(columns=LOG_COLUMNS), interactive=False,
                                      label="Response-time log (newest first)")
                clear = gr.Button("Clear chat and log")
        compare_df = gr.Dataframe(value=pd.DataFrame(columns=["model", "label", "confidence", "latency (ms)"]),
                                  interactive=False, label="All models on the last message")

        inputs = [msg, model_dd, compare, chat_state, log_state]
        outputs = [chat_html, msg, chat_state, log_state, log_df, stats, compare_df]
        msg.submit(on_send, inputs, outputs)
        send.click(on_send, inputs, outputs)
        clear.click(on_clear, None, [chat_html, chat_state, log_state, log_df, stats, compare_df])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ensure_models()
    if not predictors.available():
        raise SystemExit("No trained models found. See README: train on Colab or download the weights.")
    print("Available models:", ", ".join(n for _, n in predictors.available()))
    # Load and warm up every model BEFORE the first user message, so the first
    # message is not slowed down by loading (a cold start would distort the latency log).
    for key, name in predictors.available():
        print(f"  loading {name} ...")
        for _ in range(3):
            get(key).predict_one("warm-up message")
    build_demo().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
