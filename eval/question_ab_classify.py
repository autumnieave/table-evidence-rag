# -*- coding: utf-8 -*-
"""问题集 A/B/C 分类与 A 类 faithfulness 重算。

分类口径：

- A（研报可答）：观点、趋势、原因分析、研报明确披露的名单/预测，答案核心
  不依赖精确财报值；
- B（财报/DB 可答）：单季实际值、精确财务数字、排名、均值、比率计算；
  即使同时要求研报解释，只要精确值是硬依赖，就归 B；
- C（不确定/不可独立回答）：残句、来源质疑、图表指令、元问题，或所需字段
  在当前 MySQL schema 中不存在、研报覆盖也不确定（分部收入、电商推广费、
  出口占比等）。

用法::

    .venv\\Scripts\\python.exe eval/question_ab_classify.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "训练结果数据" / "agent_routing_full108_20261015.json"
ENTITIES = ROOT / "训练结果数据" / "l5b_measure" / "entities108.json"
FAITHFULNESS = ROOT / "训练结果数据" / "stage5_measure" / "faithfulness_refs.json"
OUT_JSON = ROOT / "训练结果数据" / "stage5_measure" / "questions_ab_classification.json"
OUT_DOC = ROOT / "docs" / "评估报告" / "问题集AB分类_20261018.md"


A_SET = {
    "B2002|1",
    "B2003|2",
    "B2004|1",
    "B2006|2",
    "B2007|2",
    "B2010|4",
    "B2011|3",
    "B2077|1",
    "B2078|1",
    "B2079|1",
    "B2080|1",
    "B2080|2",
}

A_REASON = {
    "B2002|1": "研报/政策披露的产品名单，非财报实际值",
    "B2003|2": "营收变化原因，研报观点/归因题",
    "B2004|1": "宏观资金与市场判断，观点题",
    "B2006|2": "利润涨跌原因，研报观点/归因题",
    "B2007|2": "两家公司收益率差异原因，研报归因题",
    "B2010|4": "收入趋势涨跌原因，研报观点/归因题",
    "B2011|3": "企业亏损的共同因素，研报归因题",
    "B2077|1": "种植基地与供应链策略的定性分析及长期回报判断",
    "B2078|1": "研发投入结构与新品周期的关系，研究观点题",
    "B2079|1": "大健康产品布局对营收结构的影响，研究观点题",
    "B2080|1": "中药材价格波动对不同规模企业利润的影响，研究观点题",
    "B2080|2": "未来价格趋势及判断依据，研报预测题",
}

C_REASON = {
    # 残句、元问题、来源质疑、图表操作：不是独立的研报内容题。
    "B2005|2": "残句「2025年第三季度的」，无指标/主体，不能自包含",
    "B2007|3": "只质疑数据来源可靠性，不是研报内容题",
    "B2008|2": "「他们」指代缺失，无法确定比较对象",
    "B2009|2": "残句「2025年第三季度的」，无指标/主体",
    "B2010|3": "纯图表指令，没有数据请求",
    "B2011|2": "用户给出「亏钱」定义，是澄清不是独立问题",
    "B2014|2": "只问数据来源，不构成研报内容题",
    "B2027|2": "对上一轮名单的元质询，不是新问题",
    "B2034|2": "只问数据来源，不构成研报内容题",
    "B2036|2": "只问计算口径是否正确，属校验追问",
    "B2039|2": "对名单的元质询，不是新问题",
    "B2051|2": "数据来源可靠性评估，不是研报内容题",
    "B2059|2": "数据来源可靠性评估，不是研报内容题",
    "B2067|2": "结论正确性质询，不是新问题",
    "B2073|2": "因果依据质询且指代缺失，不能独立回答",
    # 当前 DB schema 无对应字段，研报覆盖也不确定，先不计入 A/B 分母。
    "B2019|1": "配方颗粒业务收入占比不在财务表 schema；需分部报告数据",
    "B2025|1": "饮片业务收入排名无分部字段；研报只有行业趋势",
    "B2028|1": "老年病药品收入占比无分部字段；现有 SQL 只能取 ROE，不能支撑占比筛选",
    "B2029|1": "FDA 认证与海外业务收入均无结构化字段；研报仅能确认认证线索",
    "B2032|1": "电商推广费不在利润表字段；需销售费用明细",
    "B2034|1": "出口业务占比无字段；需分部/地区收入",
}

C_DATA_GAP = {
    "B2019|1",
    "B2025|1",
    "B2028|1",
    "B2029|1",
    "B2032|1",
    "B2034|1",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_items() -> List[Tuple[str, Dict[str, Any]]]:
    routing = _load_json(ROUTING)["items"]
    entities = _load_json(ENTITIES)
    qids = list(entities.keys())
    if len(routing) != len(qids):
        raise ValueError(f"路由条数 {len(routing)} != 实体条数 {len(qids)}")
    paired: List[Tuple[str, Dict[str, Any]]] = []
    for qid, item in zip(qids, routing):
        expected = str(entities[qid].get("子问题") or "")
        actual = str(item.get("问题") or "")
        if expected != actual:
            raise ValueError(f"题号错位: {qid}\n期望: {expected}\n实际: {actual}")
        paired.append((qid, item))
    return paired


def _category(qid: str) -> str:
    if qid in A_SET:
        return "A"
    if qid in C_REASON:
        return "C"
    return "B"


def _basis(qid: str, category: str, question: str) -> str:
    if category == "A":
        return A_REASON[qid]
    if category == "C":
        return C_REASON[qid]
    research_hint = any(
        key in question
        for key in (
            "研报", "观点", "评价", "原因", "为什么", "风险", "预测",
            "结合", "政策", "龙头", "核心竞争力", "解释", "驱动",
        )
    )
    if research_hint:
        return "含研报解释/筛选，但精确财务值、排名或计算是硬依赖，SQL 是前置"
    return "核心要求精确财务值、排名或计算，SQL/财报库是硬依赖"


def _sql_route(qid: str, category: str, question: str) -> str:
    if category == "A":
        return "—（研报文本；不要求 SQL）"
    if category == "C":
        if qid in C_DATA_GAP:
            return "—（需补分部/费用/地区明细；当前 schema 无字段）"
        return "—（非数据查询；需补上下文，不纳入分母）"

    tables: List[str] = []
    balance_hit = any(
        key in question
        for key in (
            "资产负债率", "总资产", "总负债", "股东权益", "应收账款", "存货",
            "货币资金", "未分配利润", "流动比率", "流动资产", "流动负债", "短期借款",
        )
    )
    cash_hit = any(key in question for key in ("现金流", "经营活动", "投资活动", "筹资活动"))
    income_hit = any(
        key in question
        for key in (
            "营收", "营业收入", "收入", "净利润", "利润总额", "研发费用", "销售费用",
            "管理费用", "营业成本", "营业总成本", "毛利率", "净利率", "收益率", "ROE",
            "扣非", "增长率", "同比", "复合增长", "费用率",
        )
    )
    if balance_hit:
        tables.append("balance_sheet")
    if cash_hit:
        tables.append("cash_flow_sheet")
    if income_hit or not tables:
        tables.append("core_performance_indicators_sheet")
        tables.append("income_sheet")
    if any(key in question for key in ("公司", "企业", "简称", "代码", "龙头", "名单")):
        tables.append("dim_stock_info")

    deduped: List[str] = []
    for table in tables:
        if table not in deduped:
            deduped.append(table)

    route = " + ".join(deduped) + "（按 stock_code/report_year/report_period join）"
    if any(
        key in question
        for key in (
            "排名", "top", "Top", "TOP", "前五", "前十", "最高", "最低", "均值", "平均",
            "统计", "超过", "低于", "高于", "对比", "筛选", "满足", "数量", "占比",
        )
    ):
        route += " + SQL 聚合/排序/窗口"
    return route


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "ci95": [0.0, 0.0], "n_ge_0_8": 0}
    n = len(values)
    mean = _mean(values)
    ordered = sorted(values)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    if n > 1:
        sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        sd = 0.0
        se = 0.0
    # A 类 n=12，使用 t(0.975, 11)=2.201；其他类别仅作描述统计。
    t_crit = 2.201 if n == 12 else 1.96
    return {
        "n": n,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "sd": round(sd, 6),
        "ci95": [round(mean - t_crit * se, 6), round(mean + t_crit * se, 6)],
        "n_ge_0_8": sum(1 for x in values if x >= 0.8),
    }


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def build() -> Dict[str, Any]:
    paired = _paired_items()
    faith = {row["qid"]: row for row in _load_json(FAITHFULNESS)["per_question"]}
    items: List[Dict[str, Any]] = []
    for qid, item in paired:
        question = str(item.get("问题") or "")
        category = _category(qid)
        frow = faith.get(qid, {})
        summary = frow.get("summary", {})
        items.append(
            {
                "qid": qid,
                "question": question,
                "category": category,
                "basis": _basis(qid, category, question),
                "sql_route": _sql_route(qid, category, question),
                "route": item.get("route"),
                "has_sql": bool(item.get("SQL")),
                "ref_count": len(item.get("引用") or []),
                "answer_len": len(str(item.get("答案") or "")),
                "faithfulness_support_rate": summary.get("support_rate"),
                "faithfulness_claims": summary.get("n_claims"),
                "faithfulness_supported": summary.get("n_supported"),
            }
        )

    counts = Counter(row["category"] for row in items)
    by_cat = {cat: _stats([row["faithfulness_support_rate"] for row in items if row["category"] == cat and row["faithfulness_support_rate"] is not None]) for cat in ("A", "B", "C")}
    by_cat["ALL"] = _stats([row["faithfulness_support_rate"] for row in items if row["faithfulness_support_rate"] is not None])
    route_by_cat = {cat: dict(Counter(row["route"] for row in items if row["category"] == cat)) for cat in ("A", "B", "C")}

    a_rows = [row for row in items if row["category"] == "A"]
    a_claims = sum(int(row["faithfulness_claims"] or 0) for row in a_rows)
    a_supported = sum(int(row["faithfulness_supported"] or 0) for row in a_rows)
    summary = {
        "n_questions": len(items),
        "categories": {cat: counts.get(cat, 0) for cat in ("A", "B", "C")},
        "faithfulness": by_cat,
        "A_claim_pool": {
            "n_claims": a_claims,
            "n_supported": a_supported,
            "pooled_rate": round(a_supported / a_claims, 6) if a_claims else 0.0,
        },
        "route_distribution_by_category": route_by_cat,
        "A_lt_20": counts.get("A", 0) < 20,
        "step4_status": "deferred_pending_more_A_questions" if counts.get("A", 0) < 20 else "ready",
    }
    return {"date": "2026-10-18", "summary": summary, "items": items}


def render_doc(result: Dict[str, Any]) -> str:
    s = result["summary"]
    c = s["categories"]
    f = s["faithfulness"]
    a = f["A"]
    all_f = f["ALL"]
    b = f["B"]
    cc = f["C"]
    lines: List[str] = [
        "# 问题集 A/B/C 分类（108题）",
        "",
        "日期：2026-10-18",
        "",
        "## 一、结论",
        "",
        f"- 分类结果：**A 类 {c['A']} 题 / B 类 {c['B']} 题 / C 类 {c['C']} 题**（合计 {s['n_questions']}）。",
        f"- A 类题均 faithfulness：**{a['mean']:.2%}**（中位数 {a['median']:.2%}，95% CI {a['ci95'][0]:.2%}–{a['ci95'][1]:.2%}）；A 类 claim 池化支持率 {s['A_claim_pool']['pooled_rate']:.2%}（{s['A_claim_pool']['n_supported']}/{s['A_claim_pool']['n_claims']}）。",
        f"- 对照：全量题均 {all_f['mean']:.2%}；B 类 {b['mean']:.2%}；C 类 {cc['mean']:.2%}。",
        f"- A 类支持率 ≥0.8 的题：{a['n_ge_0_8']}/{a['n']}。",
        "- **A 类少于 20 题，第四步黄金三元组暂不启动**；当前不通过硬凑分类来凑题。",
        "",
        "关键解释：A 类均值不是“研报 RAG 的真实能力”最终值。它仍使用**引用卡片 + L1 命中上下文**口径，不是 full_text 全文口径；A 类只有 12 题，CI 很宽；开放式观点题会被数字 claim 抽取规则计入部分描述性数字。该数字只能作方向性下界，不能与企业的 0.80+ 直接对标。",
        "",
        "## 二、分类口径",
        "",
        "| 类别 | 定义 | 是否计入研报 RAG faithfulness |",
        "|---|---|---|",
        "| A：研报可答 | 观点、趋势、原因分析、研报明确披露的名单/预测；答案核心不依赖精确财报值 | 计入 |",
        "| B：财报/DB 可答 | 单季实际值、精确财务数字、排名、均值、比率计算；含研报解释但精确值仍是硬依赖 | 剔除，路由到 SQL/财报路 |",
        "| C：不确定/不可独立回答 | 残句、来源质疑、图表指令、元问题，或当前 schema 无字段且研报覆盖不确定 | 暂不计入，单独列出 |",
        "",
        "判定顺序：先排除不能自包含的追问/元问题 → 再看是否要求精确单季值、排名或计算 → 最后才归 A。混合题只要 SQL 是硬依赖，就归 B。",
        "",
        "## 三、A 类 faithfulness 明细",
        "",
        "| 题号 | 支持率 | claims | supported | 分类依据 |",
        "|---|---:|---:|---:|---|",
    ]
    a_items = [row for row in result["items"] if row["category"] == "A"]
    for row in a_items:
        lines.append(
            f"| {row['qid']} | {row['faithfulness_support_rate']:.2%} | {row['faithfulness_claims']} | {row['faithfulness_supported']} | {_md_escape(row['basis'])} |"
        )
    lines += [
        "",
        "A 类均值 41.00%，低于全量 44.05%。这不是“剔除 B 后能力提升”的证据，反而说明当前 A 类题以开放式观点为主，且引用材料口径对观点题的覆盖率不足。不能把 41.00% 写成“研报 RAG 已达到的真实能力”。",
        "",
        "## 四、C 类说明",
        "",
        f"C 类共 {c['C']} 题，分两种：",
        "",
        "- **元问题/残句/图表指令（15 题）**：不构成独立的研报内容问题，不能进入 faithfulness 分母。",
        "- **数据口径缺口（6 题）**：配方颗粒/饮片/老年病药品收入占比、FDA/海外收入、电商推广费、出口占比；当前 `database/schema.sql` 没有分部收入/费用/地区字段，研报覆盖也不确定。",
        "",
        "这 6 题不应被强行做成 B 类 SQL：B2028 的现有 SQL 只取了 ROE 前五，无法支撑“老年病药品收入占比前五”；B2019/B2032/B2034 的 SQL 也直接缺字段。正确处置是补分部数据，或在报告中明确为能力边界。",
        "",
        "## 五、B 类 SQL 路由建议",
        "",
        "按字段族归并（逐题建议见第六节）：",
        "",
        "| 路由族 | 表 | 适用题 |",
        "|---|---|---|",
        "| 收入/利润/增长率/费用/比率 | `core_performance_indicators_sheet` + `income_sheet` | B2001、B2005、B2006、B2009、B2012–B2018、B2033、B2035–B2059、B2060–B2076 等 |",
        "| 资产/负债/应收/存货/货币资金/未分配利润 | `balance_sheet` | B2036、B2043、B2047、B2048、B2057、B2062、B2064、B2066、B2071、B2072、B2074 等 |",
        "| 经营/投资/筹资现金流 | `cash_flow_sheet` | B2020、B2044、B2048、B2050、B2053、B2061、B2069、B2072、B2076 等 |",
        "| 公司名/代码映射 | `dim_stock_info` | 需要公司名单、简称或代码的题 |",
        "| 排名/均值/计数/占比 | 上述表 + SQL 聚合/排序/窗口 | B2001、B2008、B2013、B2036、B2038–B2051 等 |",
        "",
        "边界：B 类只是“SQL/财报路可答”，不等于当前 SQL 已正确；本轮没有执行 SQL 正确性核验。",
        "",
        "## 六、逐题分类",
        "",
        "| 题号 | 类型 | route | 类别 | 分类依据 | SQL 路由/备注 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["items"]:
        lines.append(
            "| {qid} | {qtype} | {route} | {cat} | {basis} | {sql} |".format(
                qid=_md_escape(row["qid"]),
                qtype=_md_escape(row.get("question_type") or "—"),
                route=_md_escape(row.get("route") or "—"),
                cat=row["category"],
                basis=_md_escape(row["basis"]),
                sql=_md_escape(row["sql_route"]),
            )
        )
    lines += [
        "",
        "## 七、停止点",
        "",
        "第三步已完成，停在这里。A 类只有 12 题（<20），第四步黄金三元组不启动；若要继续，需要先补至少 8 道“研报文本可独立回答”的 A 类题，而不是把 B/C 题改标签。",
        "",
        "复现命令：`.venv\\Scripts\\python.exe eval/question_ab_classify.py`",
        "",
        "数据源：`训练结果数据/agent_routing_full108_20261015.json`、`训练结果数据/l5b_measure/entities108.json`、`训练结果数据/stage5_measure/faithfulness_refs.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    result = build()
    # question_type 从 entities 补齐，便于逐题表展示。
    entities = _load_json(ENTITIES)
    for row in result["items"]:
        row["question_type"] = entities.get(row["qid"], {}).get("问题类型")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_DOC.write_text(render_doc(result), encoding="utf-8")

    s = result["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"written: {OUT_JSON}")
    print(f"written: {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
