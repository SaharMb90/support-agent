"""Mock back-office tools the agent can call (stand-ins for real CRM / order APIs)."""
import itertools
import json
from pathlib import Path

_ticket_ids = itertools.count(5001)


class OrderTools:
    def __init__(self, orders_path: str | Path):
        self.orders = json.loads(Path(orders_path).read_text(encoding="utf-8"))

    def lookup_order(self, order_id: str) -> dict | None:
        order = self.orders.get(order_id)
        return {"order_id": order_id, **order} if order else None

    def cancel_order(self, order_id: str) -> dict:
        order = self.orders[order_id]
        if not order["cancellable"]:
            return {"ok": False, "reason": order["status"]}
        order.update(status="cancelled", cancellable=False)
        return {"ok": True}

    def create_ticket(self, kind: str, message: str, order_id: str | None = None) -> dict:
        return {"ticket_id": f"T{next(_ticket_ids)}", "kind": kind, "order_id": order_id, "message": message}

    def handoff_to_human(self, reason: str, transcript: list[str]) -> dict:
        return {"queue": "human_support", "reason": reason, "context_messages": len(transcript)}
