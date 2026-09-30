# -*- coding: utf-8 -*-
"""L5 稳定键：`retrieval_key_index` + `retrieval_k50_probe.payload_key` 单测（纯逻辑，零外部依赖）

覆盖：缓存键格式、稳定键优先 content_hash、缺哈希时的 sha1 兜底（确定性/区分度）、
build() 的 join 与统计（含未 join 明细、题内重复），以及探针侧 payload_key 与稳定键同定义。
不依赖 Qdrant / DashScope / 网络。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.data_scripts.retrieval_k50_probe import payload_key
from tools.data_scripts.retrieval_key_index import build, cache_key, stable_key


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path, chunks_spec):
    """构造最小 labels + cache 夹具。chunks_spec = [(bid, sub_idx, rank, content_hash, text)]"""
    labels = {"questions": []}
    cache = {}
    by_q = {}
    for bid, sub_idx, rank, chash, text in chunks_spec:
        by_q.setdefault((bid, sub_idx), []).append(
            {"rank": rank, "人工判定": "完全相关", "计入Precision": True}
        )
        cache[cache_key(bid, sub_idx, rank)] = {
            "content_hash": chash,
            "chunk_index": rank,
            "来源文件": "report.md",
            "片段全文": text,
        }
    for (bid, sub_idx), chunks in by_q.items():
        labels["questions"].append(
            {
                "bid": bid, "sub_idx": sub_idx, "子问题": "q", "问题类型": "融合",
                "chunks": sorted(chunks, key=lambda c: c["rank"]),
            }
        )
    lp = _write(tmp_path / "labels.json", labels)
    cp = _write(tmp_path / "cache.json", cache)
    return lp, cp


def test_cache_key_format():
    assert cache_key("B2040", 1, 7) == "B2040|1|7"


def test_stable_key_prefers_content_hash():
    assert stable_key({"content_hash": "abc123", "来源文件": "f.md", "片段全文": "t"}) == "abc123"


def test_stable_key_fallback_is_deterministic_and_discriminative():
    a = stable_key({"content_hash": None, "来源文件": "a.md", "片段全文": "同一段文本"})
    b = stable_key({"content_hash": None, "来源文件": "a.md", "片段全文": "同一段文本"})
    c = stable_key({"content_hash": None, "来源文件": "b.md", "片段全文": "同一段文本"})
    d = stable_key({"content_hash": None, "来源文件": "a.md", "片段全文": "另一段文本"})
    assert a == b
    assert a.startswith("sha1:")
    assert a != c and a != d
    expect = "sha1:" + hashlib.sha1("a.md\n同一段文本".encode("utf-8")).hexdigest()
    assert a == expect


def test_build_joins_and_counts(tmp_path):
    lp, cp = _fixtures(
        tmp_path,
        [
            ("B001", 1, 1, "h1", "t1"),
            ("B001", 1, 2, "h2", "t2"),
            ("B002", 2, 1, "h3", "t3"),
        ],
    )
    data = build(lp, cp)
    st = data["stats"]
    assert st["题数"] == 2
    assert st["标注条目数"] == 3
    assert st["未 join 到缓存的条目"] == 0
    assert st["全局唯一键数"] == 3
    assert st["全局复用键数"] == 0
    assert st["题内重复键的题数"] == 0
    assert [it["key"] for it in data["items"]] == ["h1", "h2", "h3"]
    assert data["items"][0]["key_source"] == "content_hash"
    assert data["items"][0]["人工判定"] == "完全相关"


def test_build_reports_missing_cache_entries(tmp_path):
    lp, cp = _fixtures(tmp_path, [("B001", 1, 1, "h1", "t1")])
    cache = json.loads(cp.read_text(encoding="utf-8"))
    cache.pop("B001|1|1")
    cp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    data = build(lp, cp)
    assert data["stats"]["未 join 到缓存的条目"] == 1
    assert data["stats"]["未 join 明细"] == ["B001|1|1"]
    assert data["items"] == []


def test_build_flags_intra_question_duplicate_keys(tmp_path):
    lp, cp = _fixtures(tmp_path, [("B001", 1, 1, "same", "t1"), ("B001", 1, 2, "same", "t1")])
    data = build(lp, cp)
    assert data["stats"]["题内重复键的题数"] == 1
    assert data["stats"]["题内重复明细"] == {"B001|1": ["same"]}


def test_payload_key_matches_stable_key_definition():
    assert payload_key({"content_hash": "zzz", "content": "x", "file_path": "f.md"}) == "zzz"
    fallback = payload_key({"content": "正文", "file_path": "f.md"})
    assert fallback == stable_key({"content_hash": None, "来源文件": "f.md", "片段全文": "正文"})
