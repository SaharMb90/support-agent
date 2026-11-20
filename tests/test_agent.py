import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SupportAgent
from agent.router import Session
from agent.text import detect_language, extract_order_id, normalize

agent = SupportAgent()


def test_persian_digits_and_language():
    assert normalize("سفارش ۱۰۲۳۴") == "سفارش 10234"
    assert extract_order_id("سفارش ۱۰۲۳۴ کجاست") == "10234"
    assert detect_language("سفارشم کجاست") == "fa" and detect_language("where is it") == "en"


def test_multi_turn_asks_for_order_id_then_answers():
    s = Session()
    first = agent.handle("where is my order?", s)
    assert first["route"] == "auto" and not first["actions"]
    second = agent.handle("10234", s)
    assert second["intent"] == "order_status" and "10234" in second["reply"]


def test_unknown_order_hands_off():
    assert agent.handle("where is order 99999?")["route"] == "handoff"


def test_angry_customer_hands_off():
    assert agent.handle("this is ridiculous, cancel 10234")["route"] == "handoff"


def test_cancel_rules_respected():
    assert "can't be cancelled" in agent.handle("cancel order 10891")["reply"]


def test_persian_reply_for_persian_message():
    r = agent.handle("سفارش ۱۰۸۹۱ کجاست")
    assert r["language"] == "fa" and "تحویل شده" in r["reply"]
