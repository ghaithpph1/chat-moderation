"""
Model 1 (rule baseline, area B): lexicon matching with minimum edit distance.

A message is "offensive" if one of its tokens is close to a word in a list of
bad words. "Close" = small Levenshtein (minimum edit) distance, so misspellings
such as "stuupid" or "idoit" are still caught. A '*' in the message matches any
letter for free, so "f*ck" or "sh*t" are caught too.

Like non-word spelling correction, we only fuzzy-match tokens that are NOT
known, correctly spelled words. Known words (seen often in the training data)
must match exactly. Otherwise ordinary words such as "rocks" would be one
edit away from a bad word and flagged.

Run:  python -m src.lexicon      (builds models/lexicon.txt and models/known_words.txt)
"""
import math
import re
import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache

import config
from src.normalize import normalize, tokenize

WORD_RE = re.compile(r"[a-z*']+")


# ----------------------------------------------------------------------------
# Minimum edit distance (dynamic programming, as in the course)
# ----------------------------------------------------------------------------
def edit_distance(a: str, b: str, max_d: int = 99, wildcard: str = "*") -> int:
    """Minimum edit distance with the Damerau extension (optimal string alignment).

    insert / delete / substitute cost 1, and swapping two neighbouring letters
    ("idoit" -> "idiot") also costs 1 instead of 2.
    A wildcard character in `a` may be substituted by any letter of `b` at cost 0.
    Stops early (returns max_d + 1) as soon as a whole DP row is above max_d.
    """
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            ca, cb = a[i - 1], b[j - 1]
            sub_cost = 0 if (ca == cb or ca == wildcard) else 1
            best = min(prev[j] + 1,             # deletion
                       cur[j - 1] + 1,          # insertion
                       prev[j - 1] + sub_cost)  # substitution / match
            if i > 1 and j > 1 and ca != cb and ca == b[j - 2] and a[i - 2] == cb:
                best = min(best, prev2[j - 2] + 1)  # transposition of two neighbours
            cur[j] = best
        if min(cur) > max_d:
            return max_d + 1
        prev2, prev = prev, cur
    return prev[-1]


class LexiconMatcher:
    def __init__(self, words, known_words=()):
        self.words = sorted(set(words))
        self.word_set = set(self.words)
        self.known = set(known_words) - self.word_set
        self.by_length = defaultdict(list)
        for w in self.words:
            self.by_length[len(w)].append(w)
        self._match = lru_cache(maxsize=200_000)(self._match_uncached)

    def allowed_distance(self, token: str) -> int:
        if not WORD_RE.fullmatch(token):
            return 0                       # numbers, '@user', ... : exact match only
        if "*" in token:
            return 1                       # wildcard is free, allow 1 more typo
        if token in self.known or len(token) < config.FUZZY_MIN_LEN:
            return 0                       # real word or short word: exact only ("duck" != "fuck")
        return 1 if len(token) <= 7 else 2

    def _match_uncached(self, token: str):
        """Return (lexicon_word, distance) of the closest match, or None."""
        if token in self.word_set:
            return token, 0
        max_d = self.allowed_distance(token)
        if max_d == 0 and "*" not in token:
            return None
        best = None
        for length in range(len(token) - max_d, len(token) + max_d + 1):
            for w in self.by_length.get(length, ()):
                d = edit_distance(token, w, max_d=max_d)
                if d <= max_d and (best is None or d < best[1]):
                    best = (w, d)
                    if d == 0:
                        return best
        return best

    def find(self, text: str):
        """All (token, lexicon_word, similarity) matches in a raw message."""
        hits = []
        for tok in tokenize(normalize(text)):
            m = self._match(tok)
            if m:
                word, d = m
                hits.append((tok, word, 1.0 - d / max(len(tok), len(word))))
        return hits

    @classmethod
    def load(cls):
        def read(path):
            return path.read_text(encoding="utf-8").split() if path.exists() else []

        return cls(read(config.LEXICON_PATH), read(config.KNOWN_WORDS_PATH))


# ----------------------------------------------------------------------------
# Building the lexicon
# ----------------------------------------------------------------------------
def _read_word_file(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [w.strip().lower() for w in lines if w.strip() and not w.startswith("#")]


def _download_public_list():
    with urllib.request.urlopen(config.BADWORDS_URL, timeout=20) as r:
        lines = r.read().decode("utf-8").splitlines()
    # keep single alphabetic words only (the list also has multi-word phrases)
    return [w.strip().lower() for w in lines if w.strip().isalpha()]


def _data_driven_words(train_df, top_k):
    """Fallback lexicon: words most associated with the offensive class (log-odds ratio)."""
    off, clean = Counter(), Counter()
    for text, label in zip(train_df["text"], train_df["label"]):
        toks = set(t for t in tokenize(normalize(text)) if t.isalpha() and len(t) > 2)
        (off if label == 1 else clean).update(toks)
    n_off, n_clean = sum(off.values()), sum(clean.values())
    scores = {}
    for w in set(off) | set(clean):
        if off[w] + clean[w] < 5:
            continue
        scores[w] = math.log((off[w] + 1) / (n_off + 1)) - math.log((clean[w] + 1) / (n_clean + 1))
    # keep only words clearly more typical of offensive messages (log-odds > 1, i.e. ~2.7x)
    ranked = [w for w, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 1.0]
    return ranked[:top_k]


def build_lexicon():
    from src.data import load_split

    train_df = load_split("train")
    words, sources = [], []
    if not config.NO_DOWNLOAD:
        try:
            words = _download_public_list()
            sources.append(f"{len(words)} words from the LDNOOBW public list (CC BY 4.0)")
        except Exception as e:
            print(f"Could not download the public list ({e}).")
    if not words:
        words = _data_driven_words(train_df, config.LEXICON_FALLBACK_TOP_K)
        sources.append(f"top-{config.LEXICON_FALLBACK_TOP_K} log-odds words of the training set")
        print("Fallback lexicon - CHECK these words by hand:", ", ".join(words[:40]), "...")

    if config.EXTRA_LEXICON_PATH.exists():
        extra = _read_word_file(config.EXTRA_LEXICON_PATH)
        words += extra
        sources.append(f"{len(extra)} hand-added words from {config.EXTRA_LEXICON_PATH.name}")

    words = sorted(set(words))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LEXICON_PATH.write_text("\n".join(words) + "\n", encoding="utf-8")

    # Dictionary of "correctly spelled" words = frequent alphabetic training tokens
    counts = Counter(t for text in train_df["text"] for t in tokenize(normalize(text)) if t.isalpha())
    known = sorted(w for w, c in counts.items() if c >= config.KNOWN_WORD_MIN_COUNT and w not in set(words))
    config.KNOWN_WORDS_PATH.write_text("\n".join(known) + "\n", encoding="utf-8")

    print(f"Lexicon: {len(words)} words  <-  " + " + ".join(sources))
    print(f"Known-word dictionary: {len(known)} words (seen >= {config.KNOWN_WORD_MIN_COUNT}x in training)")
    print(f"Saved to {config.LEXICON_PATH} and {config.KNOWN_WORDS_PATH}")


def main():
    build_lexicon()
    matcher = LexiconMatcher.load()
    print("\nMinimum edit distance examples:")
    for a, b in [("stuupid", "stupid"), ("idoit", "idiot"), ("f*ck", "fuck"), ("duck", "fuck")]:
        print(f"  d({a!r}, {b!r}) = {edit_distance(a, b)}")
    print("\nMatcher examples:")
    for msg in ["you are sooo stuupid", "what an idoit", "f*ck this lag", "nice duck hunt", "good game everyone"]:
        print(f"  {msg!r:28} -> {matcher.find(msg)}")


if __name__ == "__main__":
    main()
