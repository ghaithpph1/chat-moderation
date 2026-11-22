"""
Model 2 (classical baseline, area B): TF-IDF features + logistic regression.

TF-IDF turns each message into a sparse vector: words (and word pairs) that
are frequent in this message but rare in the corpus get high weights.
Logistic regression then learns one weight per feature.
class_weight="balanced" compensates for having fewer offensive examples.
The regularization strength C is chosen on the validation set.

Run:  python -m src.tfidf_model
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src.data import load_split
from src.normalize import normalize, tokenize
from src.utils import classification_metrics, set_seed


def make_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                                  lowercase=False, ngram_range=config.TFIDF_NGRAMS,
                                  min_df=config.TFIDF_MIN_DF, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                   random_state=config.SEED)),
    ])


def main():
    set_seed(config.SEED)
    train, val = load_split("train"), load_split("validation")

    best_f1, best_model, best_C = -1.0, None, None
    print(f"{'C':>6}  val macro-F1")
    for C in config.TFIDF_C_GRID:
        model = make_pipeline(C).fit(train["text"], train["label"])
        f1 = classification_metrics(val["label"], model.predict(val["text"]))["macro_f1"]
        print(f"{C:>6}  {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_model, best_C = f1, model, C

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.TFIDF_PATH)
    n_features = len(best_model.named_steps["tfidf"].vocabulary_)
    print(f"\nBest C = {best_C} (val macro-F1 {best_f1:.4f}), {n_features} features")

    # The most "offensive" n-grams according to the model (nice for the slides)
    vocab = best_model.named_steps["tfidf"].get_feature_names_out()
    coefs = best_model.named_steps["clf"].coef_[0]
    top = coefs.argsort()[::-1][:15]
    print("Top offensive features:", ", ".join(vocab[i] for i in top))
    print(f"Saved to {config.TFIDF_PATH}")


if __name__ == "__main__":
    main()
