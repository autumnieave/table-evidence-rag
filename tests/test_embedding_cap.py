# -*- coding: utf-8 -*-
"""嵌入前字符兜底（`cap_embedding_texts`）单元测试：纯逻辑、零外部依赖。"""

from embeddings.client import cap_embedding_texts


def test_cap_truncates_over_long_texts():
    texts, n = cap_embedding_texts(["abc", "x" * 10], 5)
    assert texts == ["abc", "xxxxx"]
    assert n == 1


def test_cap_disabled_when_max_is_zero_or_negative():
    assert cap_embedding_texts(["x" * 100], 0) == (["x" * 100], 0)
    assert cap_embedding_texts(["x" * 100], -1) == (["x" * 100], 0)


def test_cap_keeps_exact_boundary_and_empty():
    texts, n = cap_embedding_texts(["", "12345"], 5)
    assert texts == ["", "12345"] and n == 0
