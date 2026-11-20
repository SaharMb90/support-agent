# Bilingual Customer-Support Agent (English / Persian)

A small, production-style support agent that reads customer chat messages in **English or Persian**,
detects what the customer wants, **calls back-office tools** (order lookup, cancellation, ticketing),
answers simple requests automatically, and **hands hard or sensitive cases to a human** with the full context.

Built as a compact demo of the things that matter when AI runs in real operations:
routing, tool use, safe fallbacks, latency and measurable accuracy.

## How it works

```mermaid
flowchart LR
    M[Customer message] --> N[Normalise<br/>Persian digits & letters]
    N --> C[Intent classifier<br/>TF-IDF char+word n-grams<br/>+ logistic regression]
    C --> R{Router}
    R -- human request / angry /<br/>low confidence --> H[Handoff to human<br/>with transcript]
    R -- needs order no. --> Q[Ask for order number<br/>remembers intent]
    R -- order intents --> T[Tools: lookup_order,<br/>cancel_order, create_ticket]
    R -- billing / tech --> K[create_ticket]
    T --> D[Grounded reply<br/>template EN/FA]
    K --> D
    D --> L[Optional LLM rewrite<br/>facts kept, 4s timeout,<br/>falls back to template]
```

- **8 intents:** order status, refund, cancel, change address, billing, technical problem, greeting, human agent.
- **Multi-turn:** if the customer says "where is my order?" the agent asks for the number and continues the same task.
- **Business rules:** shipped orders can't be cancelled; unknown order numbers go to a human.
- **Safety-first routing:** explicit human requests, angry messages and low-confidence predictions are escalated rather than guessed.
- **Optional LLM:** set `LLM_API_KEY` (any OpenAI-compatible endpoint) to make replies more natural. The LLM only rewrites a reply that is already grounded in tool results, so it cannot invent order data, and the agent keeps working if the LLM is slow or down.
- **Every response reports** intent, confidence, route, tools called and per-stage latency.

## Results (held-out test set, CPU, no LLM)

| Metric | Value |
|---|---|
| Intent accuracy (30 test messages) | 96.7% (EN 100%, FA 92.3%) |
| Routing accuracy (auto vs. human handoff) | 100% |
| Latency per message, p50 / p95 | ~1.2 ms / ~1.3 ms |
| 5-fold CV accuracy on the 160 training messages | 60.6% |

Honest caveat: the dataset is small (20 examples per intent, half Persian), so the cross-validation score is
much lower than the test score and the test set is too small to be conclusive. The next step would be more real
customer messages or an LLM / multilingual-embedding classifier as a fallback for low-confidence cases.
Full report: [`results/report.md`](results/report.md).

## Run it

```bash
pip install -r requirements.txt
python data/build_data.py        # regenerate dataset (already included)
python eval.py                   # metrics -> results/
pytest -q                        # 6 tests
uvicorn app:app --reload         # chat UI at http://127.0.0.1:8000
# optional LLM:
export LLM_API_KEY=... LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4o-mini
```

API: `POST /chat {"session_id": "abc", "message": "سفارش ۱۰۲۳۴ کجاست؟"}`

Example response:

```json
{"reply": "سفارش 10567 لغو شد. مبلغ ظرف ۳ تا ۵ روز به کارت شما برمی‌گردد.",
 "intent": "cancel_order", "confidence": 0.941, "language": "fa", "route": "auto",
 "actions": [{"tool": "lookup_order", "order_id": "10567", "found": true}, {"tool": "cancel_order", "ok": true}],
 "latency_ms": {"classify": 2.7, "tools": 1.21, "respond": 0.0, "total": 3.92}}
```

Demo orders: `10234` (shipped), `10567` (processing, cancellable), `10891` (delivered).

## Structure

```
agent/
  text.py        Persian normalisation, language detection, order-ID extraction, anger cues
  classifier.py  bilingual intent classifier
  tools.py       mock order / ticket / handoff tools (swap for real APIs)
  responses.py   EN/FA reply templates
  llm.py         optional OpenAI-compatible rewrite layer with timeout + fallback
  router.py      the agent: classify -> route -> tools -> reply, with latency
app.py           FastAPI service + chat UI (static/index.html)
eval.py          evaluation script
tests/           pytest suite
```

Author: Sahar Mirzabaki · [GitHub](https://github.com/SaharMb90)
