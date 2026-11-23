"""Optional LLM adjudicator for ambiguous candidate pairs.

The resolver sends pairs whose model score falls in an ambiguous band to an
adjudicator. The adjudicator only sees an evidence card: names, tickers, exchanges,
dates, the gap, the price change across the gap, the volume ratio and the overlap
correlation. It never sees ground truth.

`ChatAdapter` is the only thing the adjudicator knows about. `OpenAICompatibleChat`
speaks the common /chat/completions JSON shape over httpx, so it works with a local
server or a hosted endpoint. Configure with environment variables:

  SECMASTER_LLM_BASE_URL   e.g. http://localhost:8000/v1
  SECMASTER_LLM_MODEL      model name the server expects
  SECMASTER_LLM_API_KEY    only if the server needs one

Tests, CI and the committed results use `mock_transport()`: a deterministic stand-in
that reads the same evidence card and answers in the same JSON shape. Its numbers
describe the mock, not any real LLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Callable, Protocol

from .names import name_similarity

SYSTEM_PROMPT = (
    "You are a security-master analyst. A data vendor ended one security id and started "
    "another. Decide whether the new listing is the same company continuing (same entity) "
    "or a different company (for example a new company that reused the ticker, a merger "
    "target, or a spun-off subsidiary). Weigh names, tickers, exchange, the gap in days, the "
    "price change across the gap relative to normal daily moves, the volume level and the "
    "return correlation while both traded. Answer with JSON only."
)

CARD_TEMPLATE = (
    "Listing A (ends): ticker {old_ticker}, name \"{old_name}\", exchange {old_exchange}, "
    "last trade {old_date}\n"
    "Listing B (starts): ticker {new_ticker}, name \"{new_name}\", exchange {new_exchange}, "
    "first trade {new_date}\n"
    "Gap: {gap} business days\n"
    "Price change from A's last close to B's first close: {jump_pct:+.1f}% "
    "(z = {z:+.1f} against A's daily volatility over the gap)\n"
    "Median volume B / A: {volume_ratio:.2f}\n"
    "Return correlation while both traded: {corr}\n"
    'Reply with JSON only: {{"same_entity": true | false, "confidence": <0 to 1>, '
    '"reason": "<one sentence>"}}'
)


def evidence_card(row, dates) -> str:
    return CARD_TEMPLATE.format(
        old_ticker=row.old_ticker, old_name=row.old_name, old_exchange=row.old_exchange,
        old_date=dates[int(row.old_last)].date(), new_ticker=row.new_ticker,
        new_name=row.new_name, new_exchange=row.new_exchange,
        new_date=dates[int(row.new_first)].date(), gap=int(row.raw_gap),
        jump_pct=100 * row.jump_pct, z=row.z, volume_ratio=row.volume_ratio,
        corr=f"{row.overlap_corr:.2f}" if row.has_overlap_corr else "n/a")


class ChatAdapter(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatibleChat:
    """Minimal /chat/completions adapter with retry on 429 and 5xx."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0, transport=None,
                 max_retries: int = 2, sleep: Callable[[float], None] = time.sleep):
        import httpx

        self.base_url = (base_url or os.environ.get("SECMASTER_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("SECMASTER_LLM_MODEL", "")
        if not self.base_url or not self.model:
            raise RuntimeError("set SECMASTER_LLM_BASE_URL and SECMASTER_LLM_MODEL")
        key = api_key if api_key is not None else os.environ.get("SECMASTER_LLM_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(timeout=timeout, headers=headers, transport=transport)
        self.max_retries = max_retries
        self.sleep = sleep
        self.retries = 0

    def complete(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 120}
        for attempt in range(self.max_retries + 1):
            r = self.http.post(f"{self.base_url}/chat/completions", json=body)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                self.retries += 1
                self.sleep(0.5 * 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("unreachable")  # pragma: no cover


def parse_verdict(text: str) -> tuple[bool | None, float, str]:
    """(same_entity or None if unparseable, confidence, reason)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            same = obj.get("same_entity")
            if isinstance(same, bool):
                conf = float(obj.get("confidence", 0.6))
                return same, min(max(conf, 0.0), 1.0), str(obj.get("reason", ""))
        except (ValueError, TypeError):
            pass
    low = text.lower()
    if "not the same" in low or "different company" in low:
        return False, 0.5, text.strip()[:200]
    if "same company" in low or "same entity" in low:
        return True, 0.5, text.strip()[:200]
    return None, 0.0, ""


class LLMAdjudicator:
    def __init__(self, adapter: ChatAdapter):
        self.adapter = adapter
        self.calls = 0
        self.parse_failures = 0

    def judge(self, card: str) -> tuple[bool | None, float, str]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": card}]
        self.calls += 1
        verdict = parse_verdict(self.adapter.complete(msgs))
        if verdict[0] is None:
            self.parse_failures += 1
        return verdict


# ---------------------------------------------------------------- deterministic mock

def _field(pattern: str, text: str, default=None):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _rule_judge(card: str) -> tuple[bool, float, str]:
    """Reads the card the way a careful analyst might. Fixed rules, no ground truth."""
    tick_a = _field(r"Listing A \(ends\): ticker (\S+),", card, "")
    tick_b = _field(r"Listing B \(starts\): ticker (\S+),", card, "")
    name_a = _field(r'Listing A[^\n]*name "([^"]*)"', card, "")
    name_b = _field(r'Listing B[^\n]*name "([^"]*)"', card, "")
    ex_a = _field(r"Listing A[^\n]*exchange (\w+)", card, "")
    ex_b = _field(r"Listing B[^\n]*exchange (\w+)", card, "")
    gap = int(_field(r"Gap: (-?\d+)", card, "0"))
    pct = float(_field(r"close: ([+-]?\d+\.?\d*)%", card, "0"))
    z = float(_field(r"z = ([+-]?\d+\.?\d*)", card, "0"))
    vr = float(_field(r"Median volume B / A: (\d+\.?\d*)", card, "1"))
    corr = _field(r"traded: (-?\d\.\d+)", card)

    score, why = 0.0, []
    sim = name_similarity(name_a, name_b)
    if tick_a == tick_b:
        score += 1.0
        why.append("same ticker")
    if sim >= 0.8:
        score += 1.5
        why.append("same name")
    elif sim >= 0.5:
        score += 0.5
    elif sim < 0.3:
        score -= 1.0
        why.append("different name")
    if ex_a == ex_b:
        score += 0.3
    if corr is not None and float(corr) > 0.9:
        score += 1.5
        why.append("prices move together")
    if abs(z) < 3:
        score += 1.0
        why.append("price continues")
    elif abs(z) > 8:
        score -= 1.5
        why.append("large price jump")
    if tick_a == tick_b and sim >= 0.8 and -45 <= pct <= -8 and gap <= 2:
        score += 1.5
        why.append("drop looks like a distribution such as a spin-off")
    if vr < 0.25 or vr > 4:
        score -= 0.5
    if gap > 60:
        score -= 0.3
    same = score >= 1.5
    conf = round(1 / (1 + 2.718281828 ** (-abs(score - 1.5))), 3)
    return same, conf, ", ".join(why) or "weak evidence"


def _unit(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def mock_transport(error_rate: float = 0.08, malformed_rate: float = 0.02, seed: int = 0):
    """httpx.MockTransport answering /chat/completions deterministically from the card.

    A share `error_rate` of answers is flipped with high confidence (confident LLM
    mistakes) and `malformed_rate` come back as prose, to exercise the parser.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        card = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        same, conf, why = _rule_judge(card)
        if _unit(f"{seed}|err|{card}") < error_rate:
            same, conf, why = not same, 0.85, "confident but wrong"
        if _unit(f"{seed}|fmt|{card}") < malformed_rate:
            content = ("These look like the same company." if same
                       else "This is a different company.")
        else:
            content = json.dumps({"same_entity": same, "confidence": conf, "reason": why})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
                                                                  "content": content}}]})

    return httpx.MockTransport(handler)


def mock_adjudicator(**kw) -> LLMAdjudicator:
    chat = OpenAICompatibleChat(base_url="http://mock.local/v1", model="mock", api_key="",
                                transport=mock_transport(**kw))
    return LLMAdjudicator(chat)
