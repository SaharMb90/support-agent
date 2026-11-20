"""Evaluate the agent on the held-out bilingual test set: intent accuracy, routing accuracy, latency.

Usage: python eval.py   (writes results/report.md and results/metrics.json)
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score

from agent import SupportAgent
from agent.classifier import build_pipeline, load_jsonl
from agent.router import Session
from agent.text import detect_language

ROOT = Path(__file__).parent
test = load_jsonl(ROOT / "data/test.jsonl")
train = load_jsonl(ROOT / "data/train.jsonl")
agent = SupportAgent()

rows, lat = [], []
for _ in range(3):  # warm-up
    agent.handle("hello")
for ex in test:
    r = agent.handle(ex["text"], Session())
    lat.append(r["latency_ms"]["total"])
    rows.append({**ex, "pred_intent": r["intent"], "pred_route": r["route"], "conf": r["confidence"],
                 "lang": detect_language(ex["text"])})

def acc(rs, key):
    return sum(r[key] == r["pred_" + key] for r in rs) / len(rs)

by_lang = defaultdict(list)
for r in rows:
    by_lang[r["lang"]].append(r)

cv = cross_val_score(build_pipeline(), [r["text"] for r in train], [r["intent"] for r in train],
                     cv=StratifiedKFold(5, shuffle=True, random_state=0))
lat_sorted = sorted(lat)
metrics = {
    "test_examples": len(rows),
    "intent_accuracy": round(acc(rows, "intent"), 3),
    "routing_accuracy": round(acc(rows, "route"), 3),
    "intent_accuracy_by_language": {k: round(acc(v, "intent"), 3) for k, v in by_lang.items()},
    "train_5fold_cv_accuracy": round(float(cv.mean()), 3),
    "latency_ms_p50": round(statistics.median(lat), 2),
    "latency_ms_p95": round(lat_sorted[int(0.95 * (len(lat) - 1))], 2),
    "llm_used": agent.llm.enabled,
}
errors = [r for r in rows if r["intent"] != r["pred_intent"] or r["route"] != r["pred_route"]]

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
md = ["# Evaluation report", "", "| Metric | Value |", "|---|---|"]
md += [f"| {k} | {v} |" for k, v in metrics.items()]
md += ["", "## Errors", ""] + ([f"- `{e['text']}` — expected {e['intent']}/{e['route']}, got {e['pred_intent']}/{e['pred_route']} (conf {e['conf']})" for e in errors] or ["None."])
(ROOT / "results/report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
for e in errors:
    print("ERR", e["text"], e["intent"], e["route"], "->", e["pred_intent"], e["pred_route"], e["conf"])
