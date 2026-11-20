"""Research experiments: accuracy, calibration and confidence-based human handoff across English and Persian.

Research question
-----------------
When a support agent decides "answer automatically vs. hand off to a human" from its own confidence,
is that confidence equally trustworthy in a lower-resource language (Persian) as in English?

Experiments (all on the pooled 190-message bilingual dataset, repeated stratified 5-fold CV x 10 seeds):
  E1  Model ablation: features (char / word / char+word), classifier (LogReg / Complement NB),
      and Persian normalisation on/off.
  E2  Calibration per language: Expected Calibration Error (ECE), Brier score, reliability diagrams.
  E3  Selective prediction: risk-coverage curves and AURC per language; what a fixed handoff
      threshold means for each language (coverage, error on auto-answered messages).
  E4  Cross-lingual transfer: train on one language, test on the other.
  E5  Learning curve: accuracy per language vs. amount of training data.
  E6  Per-language thresholds: the threshold each language needs so that auto-answered
      messages stay under a target error rate, and the coverage that leaves.

Usage: python research/experiments.py   -> results/research/
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.classifier import load_jsonl  # noqa: E402
from agent.text import detect_language, normalize  # noqa: E402

OUT = ROOT / "results/research"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(10)
THRESHOLD = 0.30  # handoff threshold used by the deployed agent

rows = load_jsonl(ROOT / "data/train.jsonl") + load_jsonl(ROOT / "data/test.jsonl")
X = np.array([r["text"] for r in rows], dtype=object)
y = np.array([r["intent"] for r in rows])
lang = np.array([detect_language(t) for t in X])
LABELS = sorted(set(y))


# ---------------------------------------------------------------- models
def make(features="charword", clf="lr", norm=True):
    pre = normalize if norm else None
    char = ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=pre))
    word = ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, preprocessor=pre,
                                    token_pattern=r"(?u)\b\w+\b"))
    feats = {"char": [char], "word": [word], "charword": [char, word]}[features]
    model = LogisticRegression(C=20, max_iter=3000) if clf == "lr" else ComplementNB(alpha=0.3)
    return Pipeline([("features", FeatureUnion(feats)), ("clf", model)])


MODELS = {
    "char · LogReg (deployed)": dict(features="char", clf="lr"),
    "char+word · LogReg": dict(features="charword", clf="lr"),
    "word · LogReg": dict(features="word", clf="lr"),
    "char+word · ComplementNB": dict(features="charword", clf="nb"),
    "char · LogReg · no Persian normalisation": dict(features="char", clf="lr", norm=False),
}


def oof_probs(cfg, seed, train_frac=1.0):
    """Out-of-fold class probabilities (columns ordered as LABELS)."""
    P = np.zeros((len(y), len(LABELS)))
    rng = np.random.default_rng(seed)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if train_frac < 1.0:
            tr = rng.choice(tr, size=max(len(LABELS) * 2, int(len(tr) * train_frac)), replace=False)
        m = make(**cfg).fit(X[tr], y[tr])
        cols = [LABELS.index(c) for c in m.classes_]
        P[np.ix_(te, cols)] = m.predict_proba(X[te])
    return P


# ---------------------------------------------------------------- metrics
def ece(conf, correct, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(P, yt):
    onehot = (np.array(LABELS)[None, :] == yt[:, None]).astype(float)
    return float(((P - onehot) ** 2).sum(1).mean())


def risk_coverage(conf, correct):
    order = np.argsort(-conf)
    errs = np.cumsum(~correct[order]) / np.arange(1, len(conf) + 1)
    cov = np.arange(1, len(conf) + 1) / len(conf)
    return cov, errs, float(errs.mean())  # AURC = mean selective risk over coverages


def evaluate(P):
    pred = np.array(LABELS)[P.argmax(1)]
    conf, correct = P.max(1), pred == y
    out = {"accuracy": correct.mean(), "macro_f1": f1_score(y, pred, average="macro")}
    for L in ("en", "fa"):
        m = lang == L
        out[f"acc_{L}"] = correct[m].mean()
        out[f"ece_{L}"] = ece(conf[m], correct[m])
        out[f"brier_{L}"] = brier(P[m], y[m])
        out[f"aurc_{L}"] = risk_coverage(conf[m], correct[m])[2]
        auto = m & (conf >= THRESHOLD)
        out[f"coverage@{THRESHOLD}_{L}"] = auto.sum() / m.sum()
        out[f"auto_error@{THRESHOLD}_{L}"] = 1 - correct[auto].mean() if auto.any() else float("nan")
    out["ece"] = ece(conf, correct)
    return out


def summarise(runs):
    keys = runs[0].keys()
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


fmt = lambda ms: f"{ms[0]*100:.1f} ± {ms[1]*100:.1f}"  # noqa: E731
fmt3 = lambda ms: f"{ms[0]:.3f} ± {ms[1]:.3f}"  # noqa: E731

# ---------------------------------------------------------------- E1-E3
results, main_probs = {}, None
for name, cfg in MODELS.items():
    runs, probs = [], []
    for s in SEEDS:
        P = oof_probs(cfg, s)
        probs.append(P)
        runs.append(evaluate(P))
    results[name] = summarise(runs)
    if main_probs is None:
        main_probs = probs
    print(f"{name:48s} acc {fmt(results[name]['accuracy'])}  EN {fmt(results[name]['acc_en'])}  FA {fmt(results[name]['acc_fa'])}")

main = results["char · LogReg (deployed)"]
gaps = [evaluate(P)["acc_en"] - evaluate(P)["acc_fa"] for P in main_probs]

# Reliability diagrams + risk-coverage curves (all seeds pooled)
Pall = np.vstack(main_probs)
yall, lall = np.tile(y, len(SEEDS)), np.tile(lang, len(SEEDS))
conf_all = Pall.max(1)
corr_all = np.array(LABELS)[Pall.argmax(1)] == yall
NAMES = {"en": "English", "fa": "Persian"}

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
edges = np.linspace(0, 1, 11)
for L, c in (("en", "#1f77b4"), ("fa", "#d62728")):
    m = lall == L
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = m & (conf_all > lo) & (conf_all <= hi)
        if b.sum() >= 5:
            xs.append(conf_all[b].mean()); ys.append(corr_all[b].mean())
    ax[0].plot(xs, ys, "o-", color=c, label=f"{NAMES[L]} (ECE {main[f'ece_{L}'][0]:.3f})")
    cov, err, aurc = risk_coverage(conf_all[m], corr_all[m])
    ax[1].plot(cov, err, color=c, label=f"{NAMES[L]} (AURC {main[f'aurc_{L}'][0]:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax[0].set(xlabel="Model confidence", ylabel="Actual accuracy", title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
ax[1].set(xlabel="Coverage (share answered automatically)", ylabel="Error on answered messages",
          title="Risk–coverage (selective handoff)", xlim=(0, 1), ylim=(0, None))
for a in ax:
    a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "calibration_and_risk_coverage.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- E4 cross-lingual transfer
cfg = MODELS["char · LogReg (deployed)"]
transfer = {}
for src, tgt in (("en", "fa"), ("fa", "en")):
    m = make(**cfg).fit(X[lang == src], y[lang == src])
    transfer[f"train {NAMES[src]} → test {NAMES[tgt]}"] = float((m.predict(X[lang == tgt]) == y[lang == tgt]).mean())

# ---------------------------------------------------------------- E5 learning curve
fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
curve = {L: [] for L in ("en", "fa")}
for f in fracs:
    accs = [evaluate(oof_probs(cfg, s, train_frac=f)) for s in range(5)]
    for L in curve:
        curve[L].append(float(np.mean([a[f"acc_{L}"] for a in accs])))
fig, ax = plt.subplots(figsize=(5, 3.6))
n_train = [int(len(y) * 0.8 * f) for f in fracs]
for L, c in (("en", "#1f77b4"), ("fa", "#d62728")):
    ax.plot(n_train, curve[L], "o-", color=c, label=NAMES[L])
ax.set(xlabel="Training messages (both languages)", ylabel="Accuracy", title="Learning curve", ylim=(0, 1))
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- E6 per-language thresholds
TARGET = 0.10
per_lang = {}
for L in ("en", "fa"):
    vals = []
    for P in main_probs:
        m = lang == L
        conf = P[m].max(1); correct = (np.array(LABELS)[P[m].argmax(1)] == y[m])
        best = (1.0, 0.0)  # (threshold, coverage) - fallback: answer nothing automatically
        for t in np.unique(conf):
            auto = conf >= t
            if auto.any() and 1 - correct[auto].mean() <= TARGET and auto.mean() > best[1]:
                best = (float(t), float(auto.mean()))
        vals.append(best)
    per_lang[L] = {"threshold": (float(np.mean([v[0] for v in vals])), float(np.std([v[0] for v in vals]))),
                   "coverage": (float(np.mean([v[1] for v in vals])), float(np.std([v[1] for v in vals])))}

# ---------------------------------------------------------------- report
n_en, n_fa = int((lang == "en").sum()), int((lang == "fa").sum())
md = [
    "# Research report — confidence-based handoff across English and Persian", "",
    f"Data: {len(y)} messages ({n_en} English, {n_fa} Persian), {len(LABELS)} intents. "
    f"Protocol: stratified 5-fold CV repeated over {len(SEEDS)} seeds; mean ± std across seeds. "
    f"Handoff threshold = {THRESHOLD}.", "",
    "## E1 · Model ablation (accuracy %, macro-F1 %)", "",
    "| Model | Accuracy | Macro-F1 | English | Persian |", "|---|---|---|---|---|",
]
md += [f"| {n} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['acc_en'])} | {fmt(r['acc_fa'])} |" for n, r in results.items()]
md += ["", "## E2–E3 · Calibration and selective handoff (deployed model)", "",
       "| Metric | English | Persian |", "|---|---|---|",
       f"| ECE (lower = better calibrated) | {fmt3(main['ece_en'])} | {fmt3(main['ece_fa'])} |",
       f"| Brier score | {fmt3(main['brier_en'])} | {fmt3(main['brier_fa'])} |",
       f"| AURC (lower = better ranking of its own errors) | {fmt3(main['aurc_en'])} | {fmt3(main['aurc_fa'])} |",
       f"| Coverage at threshold {THRESHOLD} (% answered automatically) | {fmt(main[f'coverage@{THRESHOLD}_en'])} | {fmt(main[f'coverage@{THRESHOLD}_fa'])} |",
       f"| Error on auto-answered messages at {THRESHOLD} (%) | {fmt(main[f'auto_error@{THRESHOLD}_en'])} | {fmt(main[f'auto_error@{THRESHOLD}_fa'])} |",
       "", f"English − Persian accuracy gap across seeds: {np.mean(gaps)*100:.1f} ± {np.std(gaps)*100:.1f} points.", "",
       "![calibration](calibration_and_risk_coverage.png)", "",
       "## E4 · Cross-lingual transfer (accuracy %)", "", "| Setting | Accuracy |", "|---|---|"]
md += [f"| {k} | {v*100:.1f} |" for k, v in transfer.items()]
md += [f"| chance (1/{len(LABELS)}) | {100/len(LABELS):.1f} |", "",
       "## E5 · Learning curve", "", "| Training messages | English | Persian |", "|---|---|---|"]
md += [f"| {n} | {curve['en'][i]*100:.1f} | {curve['fa'][i]*100:.1f} |" for i, n in enumerate(n_train)]
md += ["", "![learning curve](learning_curve.png)", "",
       f"## E6 · Per-language thresholds for ≤{TARGET*100:.0f}% error on auto-answered messages", "",
       "| Language | Threshold needed | Coverage (% answered automatically) |", "|---|---|---|"]
md += [f"| {NAMES[L]} | {fmt3(per_lang[L]['threshold'])} | {fmt(per_lang[L]['coverage'])} |" for L in ("en", "fa")]
md += [""]
(OUT / "research_report.md").write_text("\n".join(md), encoding="utf-8")
(OUT / "research_metrics.json").write_text(json.dumps(
    {"models": results, "transfer": transfer, "learning_curve": {"n_train": n_train, **curve},
     "en_minus_fa_gap": [float(np.mean(gaps)), float(np.std(gaps))], "per_language_thresholds": per_lang}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n".join(md))
