"""
Text normalization with regular expressions (technique area B).

Gamers hide offensive words on purpose: "id10t", "stuuuupid", "f.u.c.k",
"ıdiot" (Turkish keyboard), full-width letters... We undo as much of that as
we can BEFORE any model sees the text, so every model gets the same clean input.

    normalize("@USER You are sooooo STUP1D!!! http://x.co")
    -> "@user you are soo stupid!! url"
"""
import re
import unicodedata

# Turkish letters that a user might type (ı has no accent to strip, so map it by hand)
TURKISH_MAP = str.maketrans(
    {"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
     "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u"}
)
# Leetspeak: digits / symbols used as letters
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")             # "StopHate" -> "Stop Hate"
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
# single letters glued by . - _ *   e.g. "f.u.c.k", "s-h-i-t"
DOTTED_LETTERS_RE = re.compile(r"\b(?:[a-z][.\-_*]){2,}[a-z]\b")
# single letters separated by spaces (needs 4+ letters so "u r a" is left alone)
SPACED_LETTERS_RE = re.compile(r"\b(?:[a-z] ){3,}[a-z]\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")                        # "soooo" -> "soo"
TOKEN_RE = re.compile(r"@user|[a-z0-9*]+(?:'[a-z]+)?")      # '*' kept: "f*ck" is a wildcard
LEET_TOKEN_RE = re.compile(r"\S+")
WHITESPACE_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """'fück' -> 'fuck', 'ＩＤＩＯＴ' -> 'IDIOT' (NFKC folds full-width letters)."""
    text = unicodedata.normalize("NFKC", text).translate(TURKISH_MAP)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _deleet_token(token: str) -> str:
    """Replace leet characters, but only inside a word that also has letters.

    'id10t' -> 'idiot', 'sh!t' -> 'shit', but '2019' and 'wow!' stay the same.
    """
    if token == "@user":
        return token
    core, trail = re.match(r"^(.*?)([!?.,;:]*)$", token).groups()
    if re.search(r"[a-z]", core) and re.search(r"[0-9@$!]", core):
        core = "".join(LEET_MAP.get(ch, ch) for ch in core)
    return core + trail


def normalize(text: str) -> str:
    text = str(text)
    text = HASHTAG_RE.sub(lambda m: " " + CAMEL_RE.sub(" ", m.group(1)) + " ", text)
    text = _strip_accents(text).lower()
    text = URL_RE.sub(" url ", text)
    text = EMOJI_RE.sub(" ", text)
    text = DOTTED_LETTERS_RE.sub(lambda m: re.sub(r"[.\-_*]", "", m.group()), text)
    text = SPACED_LETTERS_RE.sub(lambda m: m.group().replace(" ", ""), text)
    text = LEET_TOKEN_RE.sub(lambda m: _deleet_token(m.group()), text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(normalized_text: str) -> list:
    """Word tokens used by the lexicon, TF-IDF and word2vec models."""
    return TOKEN_RE.findall(normalized_text)


if __name__ == "__main__":
    for s in ["@USER You are sooooo STUP1D!!! http://x.co", "what an id10t", "f.u.c.k this lag",
              "s h i t team", "ıdıot", "#StopTheHate now", "gg wp 2019 :)", "ＩＤＩＯＴ", "u r a noob"]:
        print(f"{s!r:45} -> {normalize(s)!r:30} {tokenize(normalize(s))}")
