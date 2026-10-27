# -*- coding: utf-8 -*-
"""API key 解析与滑动窗口限流的纯单测。"""

from __future__ import annotations

from utils.api_security import SlidingWindowRateLimiter, extract_api_key, parse_api_keys


def test_parse_api_keys() -> None:
    assert parse_api_keys(" k1, k2 ,,k3 ") == {"k1", "k2", "k3"}
    assert parse_api_keys("") == set()


def test_extract_api_key_from_header_or_bearer() -> None:
    assert extract_api_key({"x-api-key": "k1"}) == "k1"
    assert extract_api_key({"authorization": "Bearer k2"}) == "k2"
    assert extract_api_key({}) is None
    assert extract_api_key({"authorization": "Basic nope"}) is None


def test_rate_limiter_blocks_and_resets() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("k", now=100.0)[0] is True
    assert limiter.allow("k", now=101.0)[0] is True
    allowed, retry = limiter.allow("k", now=102.0)
    assert allowed is False
    assert retry > 0
    assert limiter.allow("other", now=102.0)[0] is True
    limiter.reset("k")
    assert limiter.allow("k", now=103.0)[0] is True


def test_rate_limiter_window_expires() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=10)
    assert limiter.allow("k", now=100.0)[0] is True
    assert limiter.allow("k", now=105.0)[0] is False
    assert limiter.allow("k", now=111.0)[0] is True
