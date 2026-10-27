# -*- coding: utf-8 -*-
"""生成 L5 新集合标注工作单：
1) 在新集合 `research_reports_v3_structured` 上重跑 30 题「线上口径」top-10；
2) 主表 = 30 题 × top-10（四态栏留空，盲判）；
3) 附录 = 旧标注中「新旧集合都可复现」的 26 条对照。

产物：docs/评估报告/L5新集合标注工作单_v1.md
用法：$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m tools.data_scripts.gen_l5_workorder

依赖（均不入库）：`训练结果数据/retrieval_prelabel_20260910.json`、`训练结果数据/l5b_measure/structured/`
（`label_reachability.json` / `result_structured.json` / `l5_subset_rescore_30.json`）、
`训练结果数据/l5a_compare/l5a_compare.py`（检索栈）；需 Qdrant `research_reports_v3_structured` 在线。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
L5B_DIR = ROOT / "训练结果数据" / "l5b_measure"
STRUCT = L5B_DIR / "structured"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "训练结果数据" / "l5a_compare"))

COLLECTION = "research_reports_v3_structured"
K = 10
MAX_CHARS = 2000
STATES = ["完全相关", "部分相关", "不相关", "未判定"]


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    from config.rag_config import RAGConfig
    from l5a_compare import build_stack, online_topk

    labels = json.loads((ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json").read_text(encoding="utf-8"))
    reach = json.loads((STRUCT / "label_reachability.json").read_text(encoding="utf-8"))["frags"]

    dump = STRUCT / "l5_workorder_topk.json"
    topk_by_q = {}
    if dump.is_file():
        cached = json.loads(dump.read_text(encoding="utf-8"))
        want = {"%s|%s" % (q["bid"], q["sub_idx"]) for q in labels["questions"]}
        if set(cached) == want and all(len(v) == K for v in cached.values()):
            topk_by_q = cached
            print("[wo] 命中 top-10 缓存 %s" % dump, flush=True)

    if not topk_by_q:
        cfg = RAGConfig()
        config, retriever50, reranker = build_stack(COLLECTION, cfg.RETRIEVAL_K)
        print("[wo] 集合 %s，题数 %d" % (COLLECTION, len(labels["questions"])), flush=True)
        for i, q in enumerate(labels["questions"], 1):
            qid = "%s|%s" % (q["bid"], q["sub_idx"])
            t0 = time.time()
            rows = online_topk(retriever50, reranker, config, q["子问题"], K)
            topk_by_q[qid] = [{"rank": r.get("rank"), "file_path": r.get("file_path"),
                               "content": r.get("content")} for r in rows]
            print("  [%d/%d] %s -> %d 条（%.1fs）" % (i, len(labels["questions"]), qid,
                                                     len(topk_by_q[qid]), time.time() - t0), flush=True)
        dump.write_text(json.dumps(topk_by_q, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[wo] WROTE", dump, flush=True)

    # ---- 附录：26 条「两集合都可达」 ----
    old_rows = []          # 保持 labels 顺序，仅 credit>0
    reach_i = 0
    for q in labels["questions"]:
        qid = "%s|%s" % (q["bid"], q["sub_idx"])
        for ch in q["chunks"]:
            state = ch.get("人工判定")
            if state not in ("完全相关", "部分相关"):
                continue
            fl = reach[reach_i]
            reach_i += 1
            old_rows.append({"qid": qid, "state": state, "file": ch.get("file_path"),
                             "rank_old": ch.get("rank"), "head": (ch.get("text_head") or ""),
                             "chars": ch.get("_text_full_chars"), "old_ok": fl["old"], "new_ok": fl["new"],
                             "in_new_top10": fl["new"]})
    assert reach_i == len(reach), (reach_i, len(reach))

    # 新集合里命中的 rank（用 structured 结果里的 _frag）
    scored = json.loads((STRUCT / "result_structured.json").read_text(encoding="utf-8"))
    rank_new_by_q = {pq["bid"]: pq["_frag"] for pq in scored["variants"]["baseline"]["per_question"]}

    both = [r for r in old_rows if r["old_ok"] and r["new_ok"]]

    # ---- 生成 MD ----
    L = []
    L.append("# L5 新集合标注工作单 v1（2026-10-16）")
    L.append("")
    L.append("> **用途**：为重建后的集合 `research_reports_v3_structured` 产出**可发布的 L5 检索指标**"
             "（Hit Rate@10 / Precision@10 / nDCG@10 / MRR）。")
    L.append("> **背景**：旧 30 题标注（`训练结果数据/retrieval_prelabel_20260910.json`）绑定的是 `v3_full` 的旧块文本，"
             "改成一表一 chunk 后，**56 条相关标注里有 30 条在新集合中不可逐字复现** ⇒ 旧标注不能直接复用，必须重标。")
    L.append("> **口径**：沿用 B-24B 冻结判据（`部分相关` 计 0.5；`Precision@10` 同文件最多计 2 条），"
             "**四态**：`完全相关` / `部分相关` / `不相关` / `未判定`（`未判定` = 弃权：**该条不进分母、也不计命中**）。")
    L.append("> **盲判纪律**：主表**不提供任何机器判定、不提供旧标注**；旧标注只出现在 §3 附录，供标完后对照。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 一、怎么填")
    L.append("")
    L.append("1. 逐题阅读「问题」，再逐条读该题的 top-10 片段（按 rank 1→10 的顺序，**顺序即检索顺序**）。")
    L.append("2. 对每条片段在「判定」列填 `完全相关` / `部分相关` / `不相关` / `未判定`。")
    L.append("   - **完全相关**：能直接回答该子问题（或提供其中一步所需的关键事实/数字）。")
    L.append("   - **部分相关**：与该子问题主题相关，但只提供背景/部分信息，不足以直接回答。")
    L.append("   - **不相关**：对该子问题没有回答价值。")
    L.append("   - **未判定**：无法判断 / 弃权（**后处理会把它从分母里剔除**，不会因此拉高或拉低指标）。")
    L.append("3. `计入Precision` 列**留空即可**（后处理按 B-24B 规则自动算：同一文件最多计 2 条）。")
    L.append("4. 填完把判定回填到本文件 §2 表格，或直接在文件里改（推荐直接在 Markdown 里把 `判定` 列的 `____` 替换掉）。")
    L.append("")
    L.append("## 二、待标注（30 题 × top-10 = 300 条）")
    L.append("")
    for qi, q in enumerate(labels["questions"], 1):
        qid = "%s|%s" % (q["bid"], q["sub_idx"])
        rows = topk_by_q.get(qid) or []
        L.append("### Q%02d ｜ %s（%s）" % (qi, qid, q.get("问题类型") or "-"))
        L.append("")
        L.append("**子问题**：%s" % (q.get("子问题") or "").replace("\n", " "))
        L.append("")
        L.append("| rank | 文件 | 片段（正文） | 判定 | 计入Precision |")
        L.append("| --: | :-- | :-- | :-- | :-- |")
        for r in rows:
            content = (r.get("content") or "").replace("\r", "")
            content = content.replace("|", "\\|").replace("\n", "<br>")
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS] + "…<br>**（全文 %d 字，此处截断展示）**" % len(r.get("content") or "")
            fname = pathlib.PurePath(str(r.get("file_path") or "")).name
            L.append("| %s | %s | %s | ____ | |" % (r.get("rank"), fname.replace("|", "\\|"), content))
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 三、附录：旧标注「新旧都可复现」子集（26 条）对照")
    L.append("")
    L.append("> 这 26 条是旧 56 条相关标注中、在新集合里**文本仍可复现**的部分（涉及 %d 题）。" % len({r["qid"] for r in both}))
    L.append("> 用途：标完主表后做**旧↔新一致性对照**。`新集合 rank` 为空 = 该片段在新集合的 top-10 中未出现。")
    L.append(">")
    _ref = json.loads((STRUCT / "l5_subset_rescore_30.json").read_text(encoding="utf-8")) \
        if (STRUCT / "l5_subset_rescore_30.json").is_file() else {}
    if _ref:
        L.append("> **同时可比的基线（冻结 /30 口径，只看这 26 条）**：")
        L.append("> ")
        L.append("> | 集合 | Hit Rate@10 | nDCG@10 | MRR |")
        L.append("> | :-- | --: | --: | --: |")
        L.append("> | 旧 `v3_full` | %s | %s | %s |" % (_ref["old"]["HR@10"], _ref["old"]["nDCG@10"], _ref["old"]["MRR"]))
        L.append("> | 新 `v3_structured`（重标前） | %s | %s | %s |" % (_ref["new"]["HR@10"], _ref["new"]["nDCG@10"], _ref["new"]["MRR"]))
        L.append("> ")
        L.append("> 说明：这两行是**旧标注**限制到 26 条可比片段后的结果；重新标注完成后用同一公式即可直接对比。")
    L.append("")
    L.append("| 题 | 旧判定 | 旧 rank | 文件 | 正文字数 | 片段开头（前 60 字） | 新集合 rank |")
    L.append("| :-- | :-- | --: | :-- | --: | :-- | --: |")
    for r in both:
        rn = "-"
        for f in rank_new_by_q.get(r["qid"], []):
            if f.get("rank_old") == r["rank_old"]:
                rn = f.get("rank_new") if f.get("rank_new") is not None else "-"
                break
        head = r["head"].replace("\n", " ")[:60].replace("|", "\\|")
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["qid"].replace("|", "\\|"), r["state"], r["rank_old"], pathlib.PurePath(str(r["file"])).name.replace("|", "\\|"),
            r["chars"], head, rn))
    L.append("")
    L.append("## 四、附：被丢弃的 30 条旧标注（在新集合不可复现，仅登记）")
    L.append("")
    L.append("这些片段的旧文本在「整表切片」后不再逐字存在（旧 `[TABLE_PLACEHOLDER_x]` / `会计年度:…` 序列化），"
             "标注**无法迁移**，故不进入对照表：")
    L.append("")
    L.append("| 题 | 旧判定 | 文件 | 片段开头（前 50 字） |")
    L.append("| :-- | :-- | :-- | :-- |")
    for r in old_rows:
        if r["old_ok"] and not r["new_ok"]:
            L.append("| %s | %s | %s | %s |" % (
                r["qid"].replace("|", "\\|"), r["state"], pathlib.PurePath(str(r["file"])).name.replace("|", "\\|"),
                r["head"].replace("\n", " ")[:50].replace("|", "\\|")))
    L.append("")

    out = ROOT / "docs" / "评估报告" / "L5新集合标注工作单_v1.md"
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("[wo] WROTE %s（%d 行）" % (out, len(L)), flush=True)

    n_rows = sum(len(v) for v in topk_by_q.values())
    print("[wo] 主表行数 %d；附录 26 条子集 %d；丢弃登记 %d" % (
        n_rows, len(both), sum(1 for r in old_rows if r["old_ok"] and not r["new_ok"])))
    return 0


if __name__ == "__main__":
    _utf8()
    raise SystemExit(main())
