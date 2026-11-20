"""Lightweight bilingual intent classifier (TF-IDF character n-grams + logistic regression).

Character n-grams are robust to Persian spelling variants, typos and mixed Persian/English
messages, and predict in about a millisecond on CPU. Character-only features were chosen
over character+word features because they scored higher in the research ablation
(research/experiments.py, E1).
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .text import normalize


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=normalize)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=20, max_iter=2000))])


class IntentClassifier:
    def __init__(self, train_path: str | Path):
        rows = load_jsonl(train_path)
        self.pipeline = build_pipeline().fit([r["text"] for r in rows], [r["intent"] for r in rows])
        self.labels = list(self.pipeline.classes_)

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.pipeline.predict_proba([text])[0]
        best = probs.argmax()
        return self.labels[best], float(probs[best])
