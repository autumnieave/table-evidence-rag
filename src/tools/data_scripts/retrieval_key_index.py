# -*- coding: utf-8 -*-
"""tools/data_scripts/retrieval_key_index.py —— L5 评测片段「稳定键」索引（2026-09-30）

用途：L5（代理指标层）跨轮次比对时，需要给每个已标注片段一个**跨运行稳定**的身份。
B-24A/B 的标注文件只存 `rank / file_path / text_head / _text_full_chars`，没有点 id，
轮次一变（ranks 重排）就无法对上。本脚本从 B-24B 的人工审核缓存
（`训练结果数据/retrieval_review_cache.json`，含 Qdrant payload 原文）生成稳定键。

稳定键定义：`content_hash`
- 来源＝Qdrant payload 的 `content_hash` 字段（入库时固化，与检索轮次无关）；
- 缓存缺失该字段时退回 `sha1(file_path + "\n" + 全文)`；
- **题内唯一**（实测 30 题 × 10 条全部题内唯一）；全局不唯一（同一片段被两道题同时召回），
  故比对时必须以「题」为单位。

用法：
  python -m tools.data_scripts.retrieval_key_index
  python -m tools.data_scripts.retrieval_key_index --out 训练结果数据/retrieval_keys_20260930.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = PROJECT_ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json"
DEFAULT_CACHE = PROJECT_ROOT / "训练结果数据" / "retrieval_review_cache.json"
DEFAULT_OUT = PROJECT_ROOT / "训练结果数据" / "retrieval_keys_20260930.json"


def _stdout_utf8() -> None:
    """Windows 控制台统一 UTF-8 输出"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def cache_key(bid: str, sub_idx: Any, rank: Any) -> str:
    """审核缓存条目的键（与 B-24B `retrieval_review.py` 一致）"""
    return "%s|%s|%s" % (bid, sub_idx, rank)


def _rel(path: Path) -> str:
    """仓库内路径返回相对形式，仓库外路径原样返回（便于单测用临时目录）"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def stable_key(entry: Dict[str, Any]) -> str:
    """片段稳定键：优先 Qdrant content_hash，缺失时按全文派生 sha1"""
    h = entry.get("content_hash")
    if h:
        return str(h)
    fp = str(entry.get("来源文件") or "")
    text = str(entry.get("片段全文") or "")
    return "sha1:" + hashlib.sha1((fp + "\n" + text).encode("utf-8")).hexdigest()


def build(labels_path: Path, cache_path: Path) -> Dict[str, Any]:
    """构建键索引并做完整性校验"""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    per_question: Dict[str, List[str]] = {}

    for q in labels.get("questions", []):
        bid = q["bid"]
        sub_idx = q["sub_idx"]
        qid = "%s|%s" % (bid, sub_idx)
        per_question.setdefault(qid, [])
        for c in q.get("chunks", []):
            ck = cache_key(bid, sub_idx, c.get("rank"))
            entry = cache.get(ck)
            if entry is None:
                missing.append(ck)
                continue
            key = stable_key(entry)
            per_question[qid].append(key)
            items.append(
                {
                    "cache_key": ck,
                    "bid": bid,
                    "sub_idx": sub_idx,
                    "子问题": q.get("子问题", ""),
                    "问题类型": q.get("问题类型", ""),
                    "rank": int(c["rank"]),
                    "key": key,
                    "key_source": "content_hash" if entry.get("content_hash") else "sha1",
                    "chunk_index": entry.get("chunk_index"),
                    "file_path": entry.get("来源文件", ""),
                    "full_chars": len(str(entry.get("片段全文") or "")),
                    "人工判定": c.get("人工判定"),
                    "计入Precision": c.get("计入Precision"),
                }
            )

    uniq_global = len({it["key"] for it in items})
    intra_dupes = {qid: [k for k, n in collections.Counter(ks).items() if n > 1]
                   for qid, ks in per_question.items()}
    intra_dupes = {k: v for k, v in intra_dupes.items() if v}

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_labels": _rel(labels_path),
        "source_cache": _rel(cache_path),
        "key_definition": "content_hash（Qdrant payload，入库时固化）；缺失时 sha1(file_path + \\n + 全文)",
        "stats": {
            "题数": len(per_question),
            "标注条目数": len(items),
            "未 join 到缓存的条目": len(missing),
            "未 join 明细": missing,
            "全局唯一键数": uniq_global,
            "全局复用键数": len(items) - uniq_global,
            "题内重复键的题数": len(intra_dupes),
            "题内重复明细": intra_dupes,
            "说明": "全局复用＝同一片段被两道题同时召回（正常）；题内重复＝同一题内同键多条（需人工关注）",
        },
        "items": items,
    }


def main() -> int:
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="L5 评测片段稳定键索引")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS), help="标注文件")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="B-24B 审核缓存（含全文与 content_hash）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSON 路径")
    args = ap.parse_args()

    data = build(Path(args.labels), Path(args.cache))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    st = data["stats"]
    print("[key-index] 输出 %s" % out)
    for k in ("题数", "标注条目数", "未 join 到缓存的条目", "全局唯一键数", "全局复用键数", "题内重复键的题数"):
        print("  %s：%s" % (k, st[k]))
    if st["未 join 到缓存的条目"] or st["题内重复键的题数"]:
        print("  ⚠️ 存在异常项，见 JSON 的 stats 字段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
