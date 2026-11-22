"""
Step 1 - Download the data and save fixed train / validation / test CSV files.

Dataset: OLID (OffensEval 2019, English tweets), as packaged in TweetEval
(Barbieri et al., 2020). Labels: 0 = not offensive (clean), 1 = offensive.
The official test split is our single held-out test set for ALL models.

Run:  python -m src.data
"""
import urllib.request

import pandas as pd

import config

SPLITS = ["train", "validation", "test"]


def _from_huggingface() -> dict:
    from datasets import load_dataset

    ds = load_dataset(*config.HF_DATASET)
    return {s: ds[s].to_pandas()[["text", "label"]] for s in SPLITS}


def _from_github() -> dict:
    """Fallback: the original TweetEval text files on GitHub."""
    names = {"train": "train", "validation": "val", "test": "test"}
    out = {}
    for split, gh in names.items():
        def read(kind):
            url = config.GITHUB_DATA_URL.format(split=gh, kind=kind)
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8").splitlines()

        texts, labels = read("text"), read("labels")
        labels = [int(x) for x in labels if x.strip() != ""]
        texts = texts[: len(labels)]
        assert len(texts) == len(labels), f"{split}: {len(texts)} texts vs {len(labels)} labels"
        out[split] = pd.DataFrame({"text": texts, "label": labels})
    return out


def load_split(name: str) -> pd.DataFrame:
    """Read a saved split. keep_default_na=False stops tweets like 'NA' becoming NaN."""
    df = pd.read_csv(config.DATA_DIR / f"{name}.csv", keep_default_na=False, dtype={"text": str})
    df["label"] = df["label"].astype(int)
    return df


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = _from_huggingface()
        source = "Hugging Face: " + "/".join(config.HF_DATASET)
    except Exception as e:  # no `datasets`, no network to the Hub, ...
        print(f"Hugging Face download failed ({type(e).__name__}: {e}); using GitHub files.")
        data = _from_github()
        source = "GitHub: cardiffnlp/tweeteval"

    print(f"Source: {source}\n")
    print(f"{'split':<11}{'size':>7}{'offensive':>11}{'share':>8}")
    for split in SPLITS:
        df = data[split]
        df["text"] = df["text"].astype(str)
        df.to_csv(config.DATA_DIR / f"{split}.csv", index=False)
        n_off = int((df["label"] == 1).sum())
        print(f"{split:<11}{len(df):>7}{n_off:>11}{n_off / len(df):>8.1%}")

    print("\nExamples from the training set:")
    sample = data["train"].sample(4, random_state=config.SEED)
    for text, label in zip(sample["text"], sample["label"]):
        print(f"  [{config.LABELS[label]:>9}] {text[:100]}")
    print(f"\nSaved CSV files to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
