"""E7: Is a large language model's confidence equally trustworthy in Persian and English?

The classifier study (experiments.py) asks this question for a small TF-IDF model. This script asks it
for an LLM used zero-shot as the intent classifier, with two kinds of confidence:

  verbalised   the model states its confidence (0-100) next to its answer
  consistency  share of k sampled answers (temperature 1) that agree with the majority answer

For each language it reports accuracy, Expected Calibration Error (ECE), Brier-style squared error,
risk-coverage AURC, and what the agent's handoff threshold means (coverage, error on auto-answers),
next to the classifier's numbers from experiments.py.

Usage (any OpenAI-compatible endpoint, same variables as the agent):
    export LLM_API_KEY=...  LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
    python research/llm_confidence.py --samples 5   # -> results/research/llm_report_<model>.md
    python research/llm_confidence.py --mock          # offline check of the pipeline, no API calls

Responses are cached in results/research/llm_cache.jsonl, so a rerun only calls the API for new items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.classifier import load_jsonl  # noqa: E402
from agent.text import detect_language  # noqa: E402

OUT = ROOT / "results/research"
THRESHOLD = 0.30  # same handoff threshold as the deployed agent

INTENTS = {
    "order_status": "asks where an order is or when it will arrive",
    "refund_request": "wants to return an item or get money back",
    "cancel_order": "wants to cancel an order",
    "change_address": "wants to change the delivery address",
    "billing_issue": "a problem with a charge, payment or invoice",
    "technical_problem": "a problem with the app, website, login or account",
    "greeting": "a greeting or small talk with no request",
    "human_agent": "asks to speak to a human",
}

SYSTEM = (
    "You classify customer-support messages, written in English or Persian, into exactly one intent.\n"
    "Intents:\n" + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items()) + "\n"
    'Reply with JSON only: {"intent": "<one intent name>", "confidence": <integer 0-100>}, '
    "where confidence is your probability (in percent) that the intent is correct."
)


# ------------------------------------------------------------------ LLM access
class Client:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.key = os.getenv("LLM_API_KEY")
        self.base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = "mock" if mock else os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not mock and not self.key:
            sys.exit("Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL), or run with --mock.")
        self.cache_path = OUT / "llm_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["raw"]

    def ask(self, text: str, temperature: float, sample: int) -> str:
        key = hashlib.sha256(json.dumps([self.model, SYSTEM, text, temperature, sample]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        raw = self._mock(text, sample) if self.mock else self._call(text, temperature)
        self.cache[key] = raw
        if not self.mock:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": self.model, "text": text, "raw": raw}, ensure_ascii=False) + "\n")
        return raw

    def _call(self, text: str, temperature: float) -> str:
        import httpx
        for attempt in range(3):
            try:
                r = httpx.post(f"{self.base}/chat/completions", timeout=60,
                               headers={"Authorization": f"Bearer {self.key}"},
                               json={"model": self.model, "temperature": temperature, "messages": [
                                   {"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # network or rate-limit error: retry, then give up on this item
                if attempt == 2:
                    return f"ERROR: {e}"
        return "ERROR"

    @staticmethod
    def _mock(text: str, sample: int) -> str:
        """Deterministic fake answers, only for checking the pipeline offline. Not a result."""
        h = int(hashlib.md5(f"{text}|{sample}".encode()).hexdigest(), 16)
        intent = list(INTENTS)[h % len(INTENTS)]
        return json.dumps({"intent": intent, "confidence": 40 + h % 60})


def parse(raw: str) -> tuple[str | None, float | None]:
    """Extract (intent, confidence in [0,1]) from a model reply; (None, None) if unusable."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None, None
    intent = str(obj.get("intent", "")).strip()
    if intent not in INTENTS:
        return None, None
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return intent, None
    return intent, min(max(conf / 100.0, 0.0), 1.0)


# ------------------------------------------------------------------ metrics
def ece(conf, correct, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def aurc(conf, correct):
    order = np.argsort(-conf, kind="stable")
    risks = np.cumsum(~correct[order]) / np.arange(1, len(conf) + 1)
    return float(risks.mean())


TARGET_ERROR = 0.10  # per-language threshold analysis: keep auto-answer error at or below this


def threshold_for_target(conf, correct, target=TARGET_ERROR):
    """Lowest threshold whose auto-answered messages have error <= target, maximising coverage.

    Returns (threshold, coverage). If no threshold reaches the target, returns (None, 0.0):
    every message would have to go to a human.
    """
    best_t, best_cov = None, 0.0
    for t in np.unique(conf):
        auto = conf >= t
        if auto.any() and 1 - correct[auto].mean() <= target and auto.mean() > best_cov:
            best_t, best_cov = float(t), float(auto.mean())
    return best_t, best_cov


def summary(conf, correct, valid):
    """Metrics for one language. Unusable replies count as wrong with confidence 0."""
    conf = np.where(valid, conf, 0.0)
    correct = correct & valid
    auto = conf >= THRESHOLD
    return {
        "n": int(len(conf)),
        "invalid_replies": int((~valid).sum()),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece": ece(conf, correct),
        "squared_error": float(((conf - correct) ** 2).mean()),
        "aurc": aurc(conf, correct),
        f"coverage@{THRESHOLD}": float(auto.mean()),
        f"auto_error@{THRESHOLD}": float(1 - correct[auto].mean()) if auto.any() else float("nan"),
        "threshold_for_target": threshold_for_target(conf, correct)[0],
        "coverage_for_target": threshold_for_target(conf, correct)[1],
    }


def run(mock: bool, samples: int, limit: int | None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROOT / "data/train.jsonl") + load_jsonl(ROOT / "data/test.jsonl")
    if limit:
        rows = rows[:limit]
    client = Client(mock=mock)
    texts = [r["text"] for r in rows]
    gold = np.array([r["intent"] for r in rows])
    lang = np.array([detect_language(t) for t in texts])

    # verbalised confidence: one greedy answer per message
    v_pred, v_conf = [], []
    for i, t in enumerate(texts):
        intent, conf = parse(client.ask(t, temperature=0.0, sample=0))
        v_pred.append(intent)
        v_conf.append(conf if conf is not None else np.nan)
        print(f"\r verbalised {i + 1}/{len(texts)}", end="", flush=True)
    print()
    v_pred = np.array(v_pred, dtype=object)
    v_conf = np.array(v_conf, dtype=float)
    v_valid = (v_pred != None) & ~np.isnan(v_conf)  # noqa: E711
    v_correct = v_pred == gold

    out = {"model": client.model, "mock": mock, "threshold": THRESHOLD, "verbalised": {}, "consistency": {}}
    for L in ("en", "fa"):
        m = lang == L
        out["verbalised"][L] = summary(np.nan_to_num(v_conf[m]), v_correct[m], v_valid[m])

    # consistency confidence: k sampled answers, majority vote, agreement share
    if samples > 1:
        c_pred, c_conf, all_votes = [], [], []
        for i, t in enumerate(texts):
            votes = [parse(client.ask(t, temperature=1.0, sample=s))[0] for s in range(samples)]
            votes = [v for v in votes if v]
            all_votes.append(votes)
            if votes:
                top, count = Counter(votes).most_common(1)[0]
                c_pred.append(top)
                c_conf.append(count / samples)
            else:
                c_pred.append(None)
                c_conf.append(0.0)
            print(f"\r consistency {i + 1}/{len(texts)}", end="", flush=True)
        print()
        c_pred = np.array(c_pred, dtype=object)
        c_conf = np.array(c_conf, dtype=float)
        c_valid = c_pred != None  # noqa: E711
        for L in ("en", "fa"):
            m = lang == L
            out["consistency"][L] = summary(c_conf[m], (c_pred == gold)[m], c_valid[m])
    else:
        all_votes = [[] for _ in texts]

    out["_items"] = [{"text": t, "lang": str(lang[i]), "gold": str(gold[i]), "pred": v_pred[i],
                      "confidence": None if np.isnan(v_conf[i]) else float(v_conf[i]), "votes": all_votes[i]}
                     for i, t in enumerate(texts)]
    return out


def error_report(out: dict, high_conf: float = 0.9) -> str:
    """Which messages the model gets wrong, how confident it was, and which intents it confuses."""
    items = out["_items"]
    names = {"en": "English", "fa": "Persian"}
    md = [f"# E7 error analysis ({out['model']})", ""]
    if out["mock"]:
        md += ["**MOCK RUN: fake answers. These are not results.**", ""]

    md += ["## How the errors look, per language", "",
           f"| | English | Persian |", "|---|---|---|"]
    stats = {}
    for L in ("en", "fa"):
        wrong = [it for it in items if it["lang"] == L and it["pred"] != it["gold"]]
        confident = [it for it in wrong if (it["confidence"] or 0) >= high_conf]
        unanimous = [it for it in wrong if it["votes"] and len(set(it["votes"])) == 1 and it["votes"][0] == it["pred"]]
        stats[L] = (len(wrong), len(confident), len(unanimous))
    md += [f"| Wrong answers | {stats['en'][0]} | {stats['fa'][0]} |",
           f"| … stated confidence ≥ {high_conf:.0%} | {stats['en'][1]} | {stats['fa'][1]} |",
           f"| … all sampled answers agreed on the same wrong intent | {stats['en'][2]} | {stats['fa'][2]} |", ""]

    for L in ("en", "fa"):
        wrong = [it for it in items if it["lang"] == L and it["pred"] != it["gold"]]
        pairs = Counter((it["gold"], it["pred"] or "unusable reply") for it in wrong)
        md += [f"## {names[L]}: which intents are confused", "",
               "| True intent | Predicted | Count |", "|---|---|---|"]
        md += [f"| {g} | {p} | {n} |" for (g, p), n in pairs.most_common()] or ["| none | | |"]
        md += ["", f"## {names[L]}: every wrong answer (most confident first)", "",
               "| Message | True | Predicted | Stated conf. | Sampled answers |", "|---|---|---|---|---|"]
        for it in sorted(wrong, key=lambda x: -(x["confidence"] or 0)):
            votes = ", ".join(f"{k}×{v}" for k, v in Counter(it["votes"]).most_common()) or "–"
            conf = "–" if it["confidence"] is None else f"{it['confidence']:.2f}"
            text = it["text"].replace("|", "/")
            md.append(f"| {text} | {it['gold']} | {it['pred'] or 'unusable'} | {conf} | {votes} |")
        md.append("")
    return "\n".join(md)


def report(out: dict) -> str:
    clf_path = OUT / "research_metrics.json"
    clf, pl = None, None
    if clf_path.exists():
        clf_all = json.loads(clf_path.read_text(encoding="utf-8"))
        clf = clf_all["models"]["char · LogReg (deployed)"]
        pl = clf_all.get("per_language_thresholds")
    t = out["threshold"]
    md = [f"# E7 · LLM confidence across English and Persian ({out['model']})", ""]
    if out["mock"]:
        md += ["**MOCK RUN: random fake answers to test the pipeline. These numbers are not results.**", ""]
    rows = [("Accuracy (%)", "accuracy", 100), ("ECE", "ece", 1), ("AURC", "aurc", 1),
            (f"Coverage at {t} (%)", f"coverage@{t}", 100), (f"Error on auto-answers at {t} (%)", f"auto_error@{t}", 100),
            ("Unusable replies", "invalid_replies", 1),
            (f"Threshold for ≤{TARGET_ERROR * 100:.0f}% auto-answer error", "threshold_for_target", 1),
            (f"Coverage at that threshold (%)", "coverage_for_target", 100)]
    for method in ("verbalised", "consistency"):
        if not out[method]:
            continue
        md += [f"## {method.capitalize()} confidence", "", "| Metric | English | Persian |", "|---|---|---|"]
        for label, key, scale in rows:
            en, fa = out[method]["en"][key], out[method]["fa"][key]
            if key == "invalid_replies":
                fmt = lambda v: f"{v:.0f}"  # noqa: E731
            elif key == "threshold_for_target":
                fmt = lambda v: "none reaches it" if v is None else f"{v:.2f}"  # noqa: E731
            else:
                fmt = lambda v: f"{v * scale:.1f}" if scale == 100 else f"{v:.3f}"  # noqa: E731
            md.append(f"| {label} | {fmt(en)} | {fmt(fa)} |")
        md.append("")
    if clf:
        md += ["## Classifier baseline (from experiments.py, mean over seeds)", "",
               "| Metric | English | Persian |", "|---|---|---|",
               f"| Accuracy (%) | {clf['acc_en'][0] * 100:.1f} | {clf['acc_fa'][0] * 100:.1f} |",
               f"| ECE | {clf['ece_en'][0]:.3f} | {clf['ece_fa'][0]:.3f} |",
               f"| AURC | {clf['aurc_en'][0]:.3f} | {clf['aurc_fa'][0]:.3f} |",
               f"| Error on auto-answers at {t} (%) | {clf[f'auto_error@{t}_en'][0] * 100:.1f} | {clf[f'auto_error@{t}_fa'][0] * 100:.1f} |",
               *([f"| Threshold for ≤{TARGET_ERROR * 100:.0f}% auto-answer error | {pl['en']['threshold'][0]:.2f} | {pl['fa']['threshold'][0]:.2f} |",
                  f"| Coverage at that threshold (%) | {pl['en']['coverage'][0] * 100:.1f} | {pl['fa']['coverage'][0] * 100:.1f} |"]
                 if pl else []),
               "",
               "Note: the classifier is evaluated with cross-validation; the LLM is zero-shot on all messages.", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline fake answers to test the pipeline")
    ap.add_argument("--samples", type=int, default=0, help="answers per message for consistency confidence (e.g. 5)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N messages (for a cheap trial)")
    args = ap.parse_args()
    out = run(args.mock, args.samples, args.limit)
    suffix = "_mock" if args.mock else "_" + re.sub(r"[^A-Za-z0-9.-]+", "-", out["model"])
    errors_md = error_report(out)
    out.pop("_items")
    (OUT / f"llm_metrics{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = report(out)
    (OUT / f"llm_report{suffix}.md").write_text(md, encoding="utf-8")
    (OUT / f"llm_errors{suffix}.md").write_text(errors_md, encoding="utf-8")
    print(md)
    print(f"\nError analysis written to results/research/llm_errors{suffix}.md")


if __name__ == "__main__":
    main()
