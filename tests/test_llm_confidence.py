import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.llm_confidence import aurc, ece, parse, summary


def test_parse_valid_reply():
    assert parse('{"intent": "order_status", "confidence": 85}') == ("order_status", 0.85)


def test_parse_reply_with_extra_text_and_clipping():
    assert parse('Sure! {"intent": "greeting", "confidence": 140}') == ("greeting", 1.0)


def test_parse_rejects_unknown_intent_and_bad_json():
    assert parse('{"intent": "weather", "confidence": 90}') == (None, None)
    assert parse("no json here") == (None, None)


def test_perfect_calibration_has_zero_ece():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([True, True, False, False])
    assert ece(conf, correct) == 0.0


def test_aurc_is_lower_when_errors_have_lowest_confidence():
    correct = np.array([True, True, False])
    good = aurc(np.array([0.9, 0.8, 0.1]), correct)   # the error is ranked last
    bad = aurc(np.array([0.1, 0.8, 0.9]), correct)    # the error is ranked first
    assert abs(good - 1 / 9) < 1e-9 and good < bad


def test_unusable_replies_count_as_wrong():
    s = summary(np.array([0.9, 0.9]), np.array([True, True]), np.array([True, False]))
    assert s["accuracy"] == 0.5 and s["invalid_replies"] == 1
