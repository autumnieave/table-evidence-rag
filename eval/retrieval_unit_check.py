# -*- coding: utf-8 -*-
"""阶段5：确认多阶段检索的最终返回单元（B2015/B2065/B2069）。

输出每题的真实返回结构：整表数、每张表的 table_id/file/content_chars、row_hits 数、
evidence_decision 与迭代轨迹。用于回答“最终检索单元是整表还是整表+行证据”。

用法::

    $env:PYTHONPATH="src"; $env:STAGE3_COLLECTION="research_reports_v6_rowfacts"; \
        .venv\\Scripts\\python.exe -m eval.retrieval_unit_check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eval.multi_stage_eval import COLLECTION, _build_runtime

OUT = ROOT / "训练结果数据" / "stage5_measure" / "retrieval_unit_check.json"
ENTITIES = ROOT / "训练结果数据" / "l5b_measure" / "entities108.json"
TARGETS = ("B2015|1", "B2065|1", "B2069|1")


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _unit_of(hits: List[Dict[str, Any]], action: str) -> str:
    if not hits:
        return "拒绝+理由" if action == "reject_with_reason" else "空结果"
    rows = sum(len(h.get("payload", {}).get("row_hits") or []) for h in hits)
    return "整表+行证据" if rows else "整表"


def run() -> int:
    _stdout_utf8()
    cfg, emb, _qd, _reranker, retriever = _build_runtime()
    entities = json.loads(ENTITIES.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for qid in TARGETS:
        question = str((entities.get(qid) or {}).get("子问题") or "")
        vec = emb.generate_embeddings([question], text_type="query")[0]
        hits = retriever.retrieve_with_iteration(
            question, query_vector=vec, top_k=10, max_rounds=2, rerank=False
        )
        decision = (hits[0]["payload"].get("evidence_decision") or {}) if hits else {}
        trace = retriever.last_trace.as_dict() if retriever.last_trace else {}
        row_total = sum(len(h["payload"].get("row_hits") or []) for h in hits)
        item = {
            "qid": qid,
            "question": question,
            "collection": COLLECTION,
            "decision": decision,
            "returned_unit": _unit_of(hits, str(decision.get("action") or "")),
            "n_tables": len(hits),
            "n_row_hits": row_total,
            "tables": [
                {
                    "table_id": h["payload"].get("table_id"),
                    "file": Path(str(h["payload"].get("file_path") or "")).name,
                    "content_chars": len(str(h["payload"].get("content") or "")),
                    "row_hits": len(h["payload"].get("row_hits") or []),
                    "row_examples": [
                        str(r.get("content") or "")[:120]
                        for r in (h["payload"].get("row_hits") or [])[:2]
                    ],
                }
                for h in hits
            ],
            "trace": trace,
        }
        rows.append(item)
        print(
            "%s unit=%s tables=%d rows=%d decision=%s iterations=%d" % (
                qid, item["returned_unit"], item["n_tables"], item["n_row_hits"],
                decision.get("action"), len(trace.get("iterations") or []),
            ),
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"collection": COLLECTION, "targets": list(TARGETS), "items": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("[unit] WROTE %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
