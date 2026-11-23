import json

import httpx
import numpy as np
import pandas as pd
import pytest

from secmaster.adjudicator import (LLMAdjudicator, OpenAICompatibleChat, _rule_judge,
                                   evidence_card, mock_adjudicator, parse_verdict)
from secmaster.pipeline import resolve_with


def test_parse_json_and_prose():
    assert parse_verdict('{"same_entity": true, "confidence": 0.9, "reason": "x"}')[:2] == (True, 0.9)
    assert parse_verdict('Sure. {"same_entity": false, "confidence": 2}')[:2] == (False, 1.0)
    assert parse_verdict("This is a different company.")[0] is False
    assert parse_verdict("These look like the same company.")[0] is True
    assert parse_verdict("no idea")[0] is None


def _row(**kw):
    base = dict(old_ticker="ABC", old_name="Granite Energy Corp", old_exchange="XNYS",
                old_last=10, new_ticker="ABC", new_name="GRANITE ENERGY CORPORATION",
                new_exchange="XNYS", new_first=11, raw_gap=1, jump_pct=0.004, z=0.3,
                volume_ratio=1.0, overlap_corr=0.0, has_overlap_corr=0.0)
    base.update(kw)
    return pd.Series(base)


DATES = pd.bdate_range("2015-01-01", periods=40)


def test_card_and_rule_judge():
    card = evidence_card(_row(), DATES)
    assert "Gap: 1 business days" in card and "n/a" in card
    assert _rule_judge(card)[0] is True
    other = evidence_card(_row(new_ticker="XYZ", new_name="Polaris Robotics Ltd", z=-9.5,
                               jump_pct=-0.6, volume_ratio=0.1), DATES)
    assert _rule_judge(other)[0] is False


def test_mock_is_deterministic():
    card = evidence_card(_row(), DATES)
    a, b = mock_adjudicator(), mock_adjudicator()
    assert a.judge(card) == b.judge(card)
    assert a.calls == 1


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        assert json.loads(request.content)["temperature"] == 0
        content = json.dumps({"same_entity": True, "confidence": 0.8, "reason": "ok"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    chat = OpenAICompatibleChat("http://x/v1", "m", api_key="",
                                transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert LLMAdjudicator(chat).judge("card")[0] is True
    assert chat.retries == 1


def test_missing_config_raises(monkeypatch):
    monkeypatch.delenv("SECMASTER_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SECMASTER_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleChat()


def test_adjudicator_only_sees_the_band(uni, cands, scorers):
    c, first = cands
    res = resolve_with("combined_llm_mock", uni, c, first, scorers)
    p = scorers["combined"].predict(c)
    band = set(np.where((p >= 0.2) & (p <= 0.8))[0].tolist())
    assert set(res.adjudicated) <= band
    assert res.adjudication["calls"] == len(band)
