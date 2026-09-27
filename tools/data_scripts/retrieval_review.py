"""tools/data_scripts/retrieval_review.py —— B-24B 人工审核工作单（三模式）

用途（B-24B）：把 B-24A 的模型预标注（`训练结果数据/retrieval_prelabel_*.json`）
变成人工可审核的工作单，并在人工判完后写回 JSON。

三个模式：
  --dump               从预标注 JSON + Qdrant 取回片段全文，导出 CSV 工作单
  --show <bid> <rank>  单条查看：完整问题 + 片段全文 + 模型判定
  --apply <csv>        把人工填写结果写回 JSON（写前自动备份；CSV 兼容 UTF-8 / GBK）

判据（B-24B 定稿，2026-09-27）：
  1. 片段包含回答问题所需的数值/结论 → 相关；仅主题相近、泛泛介绍行业趋势 → 不相关；
     公司名/行业/期间任一不符 → 不相关；默认片段级。
  2. 「部分相关」进 Recall 分母不进分子（0.5 语义）：JSON 里以 `人工判定=部分相关` 记录，
     `人工修正` 保持 bool（完全相关=True / 不相关=False / 部分相关与未填=None）。
  3. Precision@10 同文件最多保留 2 条：按 rank 顺序，同文件第 3 条起
     `计入Precision=False`（该列在 --dump 时自动计算，与相关性无关）。

口径声明：本脚本**不代替人工判断**——`--apply` 只搬运 CSV 里人工填写的结果，
不做任何相关性推断。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LABELS = PROJECT_ROOT / "训练结果数据" / "retrieval_prelabel_20260910.json"
DEFAULT_CACHE = PROJECT_ROOT / "训练结果数据" / "retrieval_review_cache.json"
DEFAULT_QDRANT = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "research_reports_v3_full")

REVIEW_CHOICES = ("同", "相关", "部分相关", "不相关")
JUDGMENT_FULL = "完全相关"
JUDGMENT_PARTIAL = "部分相关"
JUDGMENT_NONE = "不相关"

MATCH_KEY_CHARS = 100
PRECISION_MAX_PER_FILE = 2

CSV_FIELDS = [
    "bid",
    "问题类型",
    "sub_idx",
    "排名",
    "来源文件",
    "分数",
    "模型判定",
    "模型理由",
    "全文长度",
    "该文件内序次",
    "是否计入 Precision",
    "片段全文",
    "人工判定",
    "人工备注",
]

# 写回 JSON 的字段（人工判定为权威，人工修正/计入Precision 为派生）
APPLY_FIELDS = ("人工判定", "人工修正", "计入Precision", "人工备注")

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_text_auto(path: Path) -> str:
    """读文本并自动兼容编码。

    Excel 在中文 Windows 下把 CSV 另存为「CSV（逗号分隔）」会用 GBK，
    所以 --apply 不能死认 utf-8-sig（B-24B 实际回填时踩到过）。
    """
    raw = Path(path).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法解码（已试 %s）：%s" % (" / ".join(ENCODINGS), path))


def _stdout_utf8() -> None:
    """Windows 控制台 UTF-8（与项目其它脚本同口径）"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# 纯逻辑（可离线单测）
# --------------------------------------------------------------------------
def match_key(text: str, n: int = MATCH_KEY_CHARS) -> str:
    """片段全文匹配键。

    B-24A 的 `text_head` 是 `payload.content[:400]` 的裸切（无归一化），
    所以用前 n 字符即可与 Qdrant 的 `content` 对齐。
    """
    return str(text or "")[:n]


def normalize_judgment(raw: Any, model_relevant: Optional[bool]) -> Optional[str]:
    """把 CSV 填写值归一为「人工判定」四态；空 → None。"""
    val = str(raw or "").strip()
    if not val:
        return None
    if val == "同":
        if model_relevant is True:
            return JUDGMENT_FULL
        if model_relevant is False:
            return JUDGMENT_NONE
        raise ValueError("模型判定为空，不能填「同」")
    if val in ("相关", JUDGMENT_FULL):
        return JUDGMENT_FULL
    if val == JUDGMENT_PARTIAL:
        return JUDGMENT_PARTIAL
    if val == JUDGMENT_NONE:
        return JUDGMENT_NONE
    raise ValueError("无法识别的人工判定：%r（可选 %s）" % (val, " / ".join(REVIEW_CHOICES)))


def derive_renjian_xiuzheng(judgment: Optional[str]) -> Optional[bool]:
    """人工判定 → JSON 的 `人工修正`（bool；部分相关与未填一律 None）。

    说明：`eval/retrieval_metrics.py` 只认 bool，判据 2 的 0.5 语义需在指标侧扩展
    （B-24B 后续步骤），此处保持向后兼容，权威值放 `人工判定`。
    """
    if judgment == JUDGMENT_FULL:
        return True
    if judgment == JUDGMENT_NONE:
        return False
    return None


def precision_flags(file_paths: Sequence[str], max_per_file: int = PRECISION_MAX_PER_FILE) -> List[bool]:
    """判据 3：按 rank 顺序，同文件前 max_per_file 条计入 Precision，第 3 条起 False。"""
    seen: Dict[str, int] = {}
    out: List[bool] = []
    for fp in file_paths:
        n = seen.get(fp, 0) + 1
        seen[fp] = n
        out.append(n <= max_per_file)
    return out


def file_ordinals(file_paths: Sequence[str]) -> List[int]:
    """每个片段在其所属文件内的出现序次（1 起）。"""
    seen: Dict[str, int] = {}
    out: List[int] = []
    for fp in file_paths:
        seen[fp] = seen.get(fp, 0) + 1
        out.append(seen[fp])
    return out


def build_question_rows(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一道题摊平成明细行（不含全文，全文由 Qdrant 侧补）。"""
    chunks = list(question.get("chunks") or [])
    paths = [str(c.get("file_path") or "") for c in chunks]
    flags = precision_flags(paths)
    ordinals = file_ordinals(paths)
    rows: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        model_relevant = chunk.get("judge_relevant")
        rows.append(
            {
                "bid": question.get("bid"),
                "问题类型": question.get("问题类型"),
                "sub_idx": question.get("sub_idx"),
                "排名": chunk.get("rank"),
                "来源文件": paths[idx],
                "分数": chunk.get("score"),
                "模型判定": {True: "相关", False: "不相关"}.get(model_relevant, ""),
                "模型理由": chunk.get("judge_reason") or "",
                "全文长度": chunk.get("_text_full_chars") or "",
                "该文件内序次": ordinals[idx],
                "是否计入 Precision": "是" if flags[idx] else "否",
                "片段全文": "",
                "人工判定": "",
                "人工备注": "",
                "_问": question.get("子问题") or "",
                "_text_head": chunk.get("text_head") or "",
                "_model_relevant": model_relevant,
            }
        )
    return rows
# --------------------------------------------------------------------------
# Qdrant 取全文
# --------------------------------------------------------------------------
def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """Qdrant REST POST（stdlib，无需额外依赖）"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def scroll_by_file(url: str, collection: str, file_path: str, page: int = 200) -> List[Dict[str, Any]]:
    """按 file_path 过滤，翻页取回该文件全部片段（payload 只取必要字段）"""
    points: List[Dict[str, Any]] = []
    body: Dict[str, Any] = {
        "filter": {"must": [{"key": "file_path", "match": {"value": file_path}}]},
        "limit": page,
        "with_payload": ["content", "file_path", "chunk_index", "content_hash"],
        "with_vector": False,
    }
    offset: Any = None
    while True:
        req = dict(body)
        if offset is not None:
            req["offset"] = offset
        res = _post_json("%s/collections/%s/points/scroll" % (url, collection), req)
        result = res.get("result") or {}
        pts = result.get("points") or []
        points.extend(pts)
        offset = result.get("next_page_offset")
        if not pts or offset is None:
            break
    return points


def build_fulltext_index(
    url: str,
    collection: str,
    file_paths: Sequence[str],
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """{file_path: {match_key: [片段, ...]}}（同键多片段时留列表，取用时按长度挑选）"""
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    targets = sorted({fp for fp in file_paths if fp})
    for i, fp in enumerate(targets, 1):
        bucket: Dict[str, List[Dict[str, Any]]] = {}
        for point in scroll_by_file(url, collection, fp):
            payload = point.get("payload") or {}
            content = str(payload.get("content") or "")
            if not content:
                continue
            bucket.setdefault(match_key(content), []).append(
                {
                    "content": content,
                    "chunk_index": payload.get("chunk_index"),
                    "content_hash": payload.get("content_hash"),
                }
            )
        index[fp] = bucket
        if verbose:
            print("  [%d/%d] %s -> %d 片段" % (i, len(targets), fp, len(bucket)))
    return index


def lookup_fulltext(
    index: Dict[str, Dict[str, List[Dict[str, Any]]]],
    file_path: str,
    text_head: str,
    full_chars: Any = None,
) -> Optional[Dict[str, Any]]:
    """按前 MATCH_KEY_CHARS 字符匹配；同键多条时优先取「长度 == _text_full_chars」的那条。"""
    bucket = (index.get(file_path) or {}).get(match_key(text_head)) or []
    if not bucket:
        return None
    if len(bucket) > 1 and isinstance(full_chars, int):
        for item in bucket:
            if len(item["content"]) == full_chars:
                return item
    return bucket[0]


# --------------------------------------------------------------------------
# 缓存（只存本轮 300 条，体积小）
# --------------------------------------------------------------------------
def entry_key(bid: Any, sub_idx: Any, rank: Any) -> str:
    return "%s|%s|%s" % (bid, sub_idx, rank)


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(read_text_auto(path))
    except Exception:  # noqa: BLE001
        return {}


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# --dump
# --------------------------------------------------------------------------
def cmd_dump(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    labels = json.loads(read_text_auto(labels_path))
    questions = list(labels.get("questions") or [])

    all_rows: List[Dict[str, Any]] = []
    for question in questions:
        all_rows.extend(build_question_rows(question))

    need_paths = sorted({r["来源文件"] for r in all_rows if r["来源文件"]})
    print("[dump] 预标注 %d 题 / %d 片段；需回源文件 %d 个" % (len(questions), len(all_rows), len(need_paths)))
    print("[dump] 从 Qdrant %s 集合 %s 取全文 ..." % (args.qdrant, args.collection))
    index = build_fulltext_index(args.qdrant, args.collection, need_paths, verbose=not args.quiet)

    cache: Dict[str, Any] = {}
    miss = 0
    for row in all_rows:
        hit = lookup_fulltext(index, row["来源文件"], row["_text_head"], row["全文长度"])
        if hit is None:
            miss += 1
            row["片段全文"] = ""
            row["人工备注"] = "[未匹配到全文]"
        else:
            row["片段全文"] = hit["content"]
            row["全文长度"] = len(hit["content"])
            cache[entry_key(row["bid"], row["sub_idx"], row["排名"])] = {
                "bid": row["bid"],
                "sub_idx": row["sub_idx"],
                "排名": row["排名"],
                "问题类型": row["问题类型"],
                "子问题": row["_问"],
                "来源文件": row["来源文件"],
                "分数": row["分数"],
                "模型判定": row["模型判定"],
                "模型理由": row["模型理由"],
                "chunk_index": hit.get("chunk_index"),
                "content_hash": hit.get("content_hash"),
                "片段全文": hit["content"],
            }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    if not args.no_cache:
        save_cache(Path(args.cache), cache)

    n_partial_hint = 0
    for question in questions:
        rows = [r for r in all_rows if r["bid"] == question.get("bid") and r["sub_idx"] == question.get("sub_idx")]
        n_partial_hint += sum(1 for r in rows if r["是否计入 Precision"] == "否")

    print("[dump] CSV: %s" % out_csv)
    print("[dump] 数据行 %d（表头 1 行 → 文件共 %d 行）" % (len(all_rows), len(all_rows) + 1))
    print("[dump] 全文取回 %d / 未匹配 %d" % (len(all_rows) - miss, miss))
    print("[dump] 判据3 不计入 Precision 的片段 %d 条" % n_partial_hint)
    print("[dump] 缓存: %s" % ("（已跳过）" if args.no_cache else args.cache))
    print("[dump] 填写列：人工判定（同 / 相关 / 部分相关 / 不相关）、人工备注")
    return 0 if miss == 0 else 2


# --------------------------------------------------------------------------
# --show
# --------------------------------------------------------------------------
def cmd_show(args: argparse.Namespace) -> int:
    cache = load_cache(Path(args.cache))
    key = entry_key(args.bid, args.sub_idx, args.rank)
    item = cache.get(key)
    if item is None and args.sub_idx in ("", None):
        item = next((v for k, v in cache.items() if k.startswith("%s|" % args.bid) and k.endswith("|%s" % args.rank)), None)
    if item is None:
        print("缓存里没有 %s；请先跑 --dump，或确认 bid/sub_idx/rank 是否正确" % key)
        return 2
    print("=" * 78)
    print("bid=%s  子问题=%s  第 %s 片段" % (item["bid"], item["sub_idx"], item["排名"]))
    print("题型=%s" % item["问题类型"])
    print("-" * 78)
    print("问题：%s" % item["子问题"])
    print("-" * 78)
    print("来源：%s（分数 %s，chunk_index %s）" % (item["来源文件"], item["分数"], item["chunk_index"]))
    print("模型判定：%s" % item["模型判定"])
    print("模型理由：%s" % item["模型理由"])
    print("-" * 78)
    print("片段全文（%d 字）：" % len(item["片段全文"]))
    print(item["片段全文"])
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------
# --apply
# --------------------------------------------------------------------------
def apply_csv(args: argparse.Namespace) -> Dict[str, Any]:
    labels_path = Path(args.labels)
    labels = json.loads(read_text_auto(labels_path))
    questions = list(labels.get("questions") or [])

    expected: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for question in questions:
        for chunk in question.get("chunks") or []:
            expected[entry_key(question.get("bid"), question.get("sub_idx"), chunk.get("rank"))] = (question, chunk)

    with io.StringIO(read_text_auto(Path(args.apply)), newline="") as fh:
        csv_rows = list(csv.DictReader(fh))

    if len(csv_rows) != len(expected):
        raise ValueError("CSV 数据行 %d != 预标注片段 %d，拒绝写回" % (len(csv_rows), len(expected)))

    seen: Dict[str, int] = {}
    judgments: Dict[str, Optional[str]] = {}
    for row in csv_rows:
        key = entry_key(row.get("bid"), row.get("sub_idx"), row.get("排名"))
        if key not in expected:
            raise ValueError("CSV 出现预标注里没有的行：%s" % key)
        if key in seen:
            raise ValueError("CSV 出现重复行：%s" % key)
        seen[key] = 1
        question, chunk = expected[key]
        model_relevant = chunk.get("judge_relevant")
        judgments[key] = normalize_judgment(row.get("人工判定"), model_relevant)

    unfilled = sum(1 for v in judgments.values() if v is None)
    if unfilled:
        raise ValueError("还有 %d 条未填「人工判定」，拒绝写回" % unfilled)

    changed = 0
    for key, judgment in judgments.items():
        question, chunk = expected[key]
        model_relevant = chunk.get("judge_relevant")
        chunk["人工判定"] = judgment
        chunk["人工修正"] = derive_renjian_xiuzheng(judgment)
        if (judgment == JUDGMENT_FULL) != (model_relevant is True):
            changed += 1
    for row in csv_rows:
        key = entry_key(row.get("bid"), row.get("sub_idx"), row.get("排名"))
        _q, chunk = expected[key]
        chunk["计入Precision"] = row.get("是否计入 Precision") == "是"
        note = str(row.get("人工备注") or "").strip()
        if note:
            chunk["人工备注"] = note

    stats = {
        "完全相关": sum(1 for v in judgments.values() if v == JUDGMENT_FULL),
        "部分相关": sum(1 for v in judgments.values() if v == JUDGMENT_PARTIAL),
        "不相关": sum(1 for v in judgments.values() if v == JUDGMENT_NONE),
        "与模型判定不同": changed,
        "总条数": len(judgments),
    }
    labels["人工审核"] = {
        "审核人": args.reviewer or "",
        "审核日期": args.review_date or _dt.date.today().isoformat(),
        "判据版本": "B-24B 定稿 2026-09-27（部分相关 0.5；Precision 同文件最多 2 条）",
        "统计": stats,
    }
    if not args.dry_run:
        backup = labels_path.with_name(labels_path.stem + ".backup-" + _dt.datetime.now().strftime("%Y%m%d%H%M%S") + labels_path.suffix)
        shutil.copy2(labels_path, backup)
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print("[apply] 已备份原 JSON -> %s" % backup)
        print("[apply] 已写回 -> %s" % labels_path)
    else:
        print("[apply] dry-run：未写盘")
    for k, v in stats.items():
        print("[apply] %s = %s" % (k, v))
    return stats


def cmd_apply(args: argparse.Namespace) -> int:
    apply_csv(args)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="B-24B retrieval 人工审核工作单（dump / show / apply）")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS), help="B-24A 预标注 / B-24B 终稿 JSON")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="本轮 300 条全文缓存")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true", help="导出 CSV 工作单（取回片段全文）")
    mode.add_argument("--show", nargs=2, metavar=("BID", "RANK"), help="单条查看：完整问题 + 片段全文")
    mode.add_argument("--apply", metavar="CSV", help="把人工填写结果写回 JSON（写前自动备份）")
    ap.add_argument("--out-csv", default="", help="CSV 输出路径（默认 训练结果数据/retrieval_review_<日期>.csv）")
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT, help="Qdrant REST 地址")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant 集合名")
    ap.add_argument("--sub-idx", default="", help="--show 时的子问题序号（默认自动匹配）")
    ap.add_argument("--no-cache", action="store_true", help="不写全文缓存")
    ap.add_argument("--quiet", action="store_true", help="不打印每个文件的取回进度")
    ap.add_argument("--reviewer", default="", help="--apply 时记录审核人")
    ap.add_argument("--review-date", default="", help="--apply 时记录审核日期（默认今天）")
    ap.add_argument("--dry-run", action="store_true", help="--apply 时只校验不写盘")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    _stdout_utf8()
    args = build_parser().parse_args(argv)
    if not args.out_csv:
        args.out_csv = str(PROJECT_ROOT / "训练结果数据" / ("retrieval_review_%s.csv" % _dt.date.today().strftime("%Y%m%d")))
    if args.dump:
        return cmd_dump(args)
    if args.show:
        args.bid, args.rank = args.show[0], args.show[1]
        return cmd_show(args)
    return cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())