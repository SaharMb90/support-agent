"""The support agent: classify -> decide route -> call tools -> reply, with per-stage latency."""
import time
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import IntentClassifier
from .llm import LLM
from .responses import render
from .text import detect_language, extract_order_id, is_angry

DATA = Path(__file__).resolve().parent.parent / "data"
NEEDS_ORDER = {"order_status", "refund_request", "cancel_order", "change_address"}


@dataclass
class Session:
    transcript: list[str] = field(default_factory=list)
    pending_intent: str | None = None  # waiting for an order number for this intent


class SupportAgent:
    def __init__(self, train_path=DATA / "train.jsonl", orders_path=DATA / "orders.json",
                 confidence_threshold: float = 0.30):
        from .tools import OrderTools
        self.clf = IntentClassifier(train_path)
        self.tools = OrderTools(orders_path)
        self.llm = LLM()
        self.threshold = confidence_threshold

    def handle(self, message: str, session: Session | None = None) -> dict:
        session = session or Session()
        session.transcript.append(message)
        t0 = time.perf_counter()
        lang = detect_language(message)
        order_id = extract_order_id(message)
        intent, conf = self.clf.predict(message)

        # Multi-turn: customer is answering our "what's your order number?" question.
        if session.pending_intent and order_id and (conf < 0.5 or len(message.split()) <= 3):
            intent, conf = session.pending_intent, 1.0
        t1 = time.perf_counter()

        actions, route = [], "auto"
        if intent == "human_agent" or is_angry(message) or conf < self.threshold:
            reason = "requested_human" if intent == "human_agent" else "negative_sentiment" if is_angry(message) else "low_confidence"
            actions.append({"tool": "handoff_to_human", **self.tools.handoff_to_human(reason, session.transcript)})
            route, draft = "handoff", render("handoff", lang)
        elif intent in NEEDS_ORDER and not order_id:
            session.pending_intent = intent
            draft = render("ask_order_id", lang)
        elif intent in NEEDS_ORDER:
            session.pending_intent = None
            order = self.tools.lookup_order(order_id)
            actions.append({"tool": "lookup_order", "order_id": order_id, "found": bool(order)})
            if not order:
                actions.append({"tool": "handoff_to_human", **self.tools.handoff_to_human("order_not_found", session.transcript)})
                route, draft = "handoff", render("order_not_found", lang, order_id=order_id)
            elif intent == "order_status":
                draft = render("order_status", lang, **order)
            elif intent == "cancel_order":
                res = self.tools.cancel_order(order_id)
                actions.append({"tool": "cancel_order", **res})
                draft = render("cancel_ok" if res["ok"] else "cancel_no", lang, **order)
            else:  # refund_request / change_address
                ticket = self.tools.create_ticket(intent, message, order_id)
                actions.append({"tool": "create_ticket", **ticket})
                draft = render(intent, lang, **order, ticket_id=ticket["ticket_id"])
        elif intent in ("billing_issue", "technical_problem"):
            ticket = self.tools.create_ticket(intent, message, order_id)
            actions.append({"tool": "create_ticket", **ticket})
            draft = render(intent, lang, ticket_id=ticket["ticket_id"])
        else:
            draft = render("greeting", lang)
        t2 = time.perf_counter()

        reply = self.llm.polish(message, draft)
        t3 = time.perf_counter()
        ms = lambda a, b: round((b - a) * 1000, 2)
        return {"reply": reply, "intent": intent, "confidence": round(conf, 3), "language": lang, "route": route,
                "actions": actions, "llm_used": self.llm.enabled,
                "latency_ms": {"classify": ms(t0, t1), "tools": ms(t1, t2), "respond": ms(t2, t3), "total": ms(t0, t3)}}
