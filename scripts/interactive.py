"""
scripts/interactive.py
交互式问答模式和主入口函数
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipelines.rag_pipeline import RAGPipeline
from pipelines.citation_validator import CitationValidator
from config.rag_config import RAGConfig
from memory import ConversationState

logger = logging.getLogger(__name__)


def _run_citation_validation(refs_path: str, match_mode: str = "comma") -> None:
    """对引用 JSON 文件执行 L1 引用核验（--validate-refs 入口）

    输入为 JSON 列表，每个元素为 {"paper_path": "...", "text": "..."}。
    校验完成后打印汇总，并在同目录输出 *_citation_report.json 逐条明细。

    Args:
        refs_path: 引用 JSON 文件路径
        match_mode: 数字匹配口径（raw / comma / loose）
    """
    with open(refs_path, "r", encoding="utf-8") as fh:
        references = json.load(fh)
    config = RAGConfig()
    validator = CitationValidator(
        corpus_root=config.CITATION_CORPUS_ROOT,
        match_mode=match_mode,
    )
    records = validator.check_references(references)
    summary = validator.summarize(records)

    print(f"\n引用核验完成：共 {summary['total']} 条引用")
    print(
        f"文件可溯源率：{summary['traceable']}/{summary['total']} = {summary['traceable_rate'] * 100:.1f}%"
        f"（exact {summary['exact']} + fuzzy {summary['fuzzy']}，missing {summary['missing']}）"
    )
    if summary["num_total"]:
        print(
            f"数字命中率：{summary['num_hit']}/{summary['num_total']} = {summary['num_rate'] * 100:.1f}%"
            f"（口径 {match_mode}）"
        )
        with_numbers = sum(1 for r in records if r["nums"] > 0)
        print(f"含数字引用 {with_numbers} 条：全命中 {summary['all_hit_refs']}，零命中 {summary['zero_hit_refs']}")
    for item in summary["missing_refs"]:
        print(f"[缺失] {item['paper_path']}")
    for item in summary["zero_hit_refs_detail"]:
        print(f"[零命中] {item['paper_path']} | 未命中数字: {item['unhit']}")

    out_path = os.path.splitext(refs_path)[0] + "_citation_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"records": records, "summary": summary}, fh, ensure_ascii=False, indent=2)
    print(f"逐条明细已写入：{out_path}")


def interactive_mode(pipeline):
    """
    交互式问答模式
    """
    print("\n" + "="*60)
    print("RAG金融研报问答系统")
    print("命令：")
    print("  - 直接输入问题：普通RAG检索")
    print("  - agent on   : 切换到Agent模式（支持多步推理/图表）")
    print("  - agent off  : 切换回普通RAG模式")
    print("  - multi-turn on   : 启用多轮对话模式（自动澄清缺失信息）")
    print("  - multi-turn off  : 禁用多轮对话模式")
    print("  - langchain on    : 切换到 LangChain 检索器")
    print("  - langchain off   : 切换回原有检索器")
    print("  - hybrid on       : 切换混合检索（向量 + BM25，实验）")
    print("  - hybrid off      : 切换回默认检索器")
    print("  - chain on        : 切换到 LangChain 完整链路（LCEL）")
    print("  - chain off       : 切换回手写链路")
    print("  - planner langgraph  : 切换 Agent 编排为 LangGraph StateGraph（实验）")
    print("  - planner multi-agent : 切换为 LangGraph 多 Agent 协作（规划→财务/研报→汇总，实验）")
    print("  - planner handwritten: 切回自研 Agent 循环（默认）")
    print("  - status     : 查看当前模式")
    print("  - rebuild    : 重建索引")
    print("  - quit       : 退出")
    print(' - addstock    : 增量插入个股研报（从 NEW_STOCK_DIR）')
    print(' - addindustry : 增量插入行业研报（从 NEW_INDUSTRY_DIR）')
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("感谢使用，再见！")
                break
            
            if user_input.lower() == "new":
                pipeline.reset_conversation(user_id="default")
                print("已开始新话题，您可以提问了。")
                continue

            if user_input.lower() == "rebuild":
                confirm = input("确认重建索引？这将清空现有数据 (y/n): ")
                if confirm.lower() == 'y':
                    pipeline.build_index(force_rebuild=True)
                continue

            # Agent 开关控制
            if user_input.lower() == "agent on":
                if not hasattr(pipeline, 'agent_planner') or pipeline.agent_planner is None:
                    print("⚠️ Agent 组件未初始化，请检查配置。")
                else:
                    pipeline.agent_mode_enabled = True
                    print("✅ 已切换到 Agent 模式（财务查询走原生 SQL 链路）。")
                continue
            if user_input.lower() == "agent off":
                pipeline.agent_mode_enabled = False
                print("✅ 已切换到普通 RAG 模式。")
                continue

            # 多轮对话开关
            if user_input.lower() == "multi-turn on":
                pipeline.enable_multy_turn = True
                print("✅ 已启用多轮对话模式。")
                continue
            if user_input.lower() == "multi-turn off":
                pipeline.enable_multy_turn = False
                print("✅ 已禁用多轮对话模式。")
                continue

            # LangChain 检索器开关
            if user_input.lower() == "langchain on":
                pipeline.use_langchain_retriever = True
                print("✅ 已切换到 LangChain 检索器。")
                continue
            if user_input.lower() == "langchain off":
                pipeline.use_langchain_retriever = False
                print("✅ 已切换回原有检索器。")
                continue

            # LCEL 完整链路开关
            if user_input.lower() == "chain on":
                pipeline.use_langchain_chain = True
                print("✅ 已切换到 LangChain 完整链路。")
                continue
            if user_input.lower() == "chain off":
                pipeline.use_langchain_chain = False
                print("✅ 已切换回手写链路。")
                continue

            # 混合检索开关（#8 实验）
            if user_input.lower() == "hybrid on":
                pipeline.use_hybrid_retriever = True
                print("✅ 已切换到混合检索（向量 + BM25）。")
                continue
            if user_input.lower() == "hybrid off":
                pipeline.use_hybrid_retriever = False
                print("✅ 已切换回默认检索器。")
                continue

            # Agent 编排后端切换（#9 实验：自研循环 vs LangGraph StateGraph）
            if user_input.lower() == "planner multi-agent":
                pipeline.config.AGENT_PLANNER_BACKEND = "langgraph"
                pipeline.config.AGENT_LANGGRAPH_MULTI_AGENT = True
                print("✅ 已切换 Agent 编排为 LangGraph 多 Agent 协作（规划→财务/研报→汇总，实验）。")
                continue
            if user_input.lower() == "planner langgraph":
                pipeline.config.AGENT_PLANNER_BACKEND = "langgraph"
                pipeline.config.AGENT_LANGGRAPH_MULTI_AGENT = False
                print("✅ 已切换 Agent 编排为 LangGraph（实验）。")
                continue
            if user_input.lower() == "planner handwritten":
                pipeline.config.AGENT_PLANNER_BACKEND = "handwritten"
                pipeline.config.AGENT_LANGGRAPH_MULTI_AGENT = False
                print("✅ 已切回自研 Agent 循环（默认）。")
                continue

            if user_input.lower() == "status":
                print(f"  Agent 模式: {'✅ 开启' if pipeline.agent_mode_enabled else '❌ 关闭'}")
                print(f"  多轮对话:   {'✅ 开启' if pipeline.enable_multy_turn else '❌ 关闭'}")
                print(f"  LangChain 检索器: {'✅ 使用中' if pipeline.use_langchain_retriever else '❌ 未启用'}")
                print(f"  混合检索(BM25):  {'✅ 使用中' if getattr(pipeline, 'use_hybrid_retriever', False) else '❌ 未启用'}")
                print(f"  LangChain 完整链路: {'✅ 使用中' if pipeline.use_langchain_chain else '❌ 未启用'}")
                backend = getattr(pipeline.config, "AGENT_PLANNER_BACKEND", "handwritten")
                multi = getattr(pipeline.config, "AGENT_LANGGRAPH_MULTI_AGENT", False)
                if backend == "langgraph":
                    backend_label = "LangGraph 多 Agent（实验）" if multi else "LangGraph（实验）"
                else:
                    backend_label = "自研循环（默认）"
                print(f"  Agent 编排后端:    {backend_label}")
                continue

            if user_input.lower() == "addstock":
                pipeline.add_new_stock_reports()
                print("个股研报增量插入完成")
            if user_input.lower() == "addindustry":
                pipeline.add_new_industry_reports()
                print("行业研报增量插入完成")
            
            if pipeline.agent_mode_enabled:
                try:
                    answer_dict = pipeline.agent_query(user_input, verbose=True)
                except Exception as e:
                    logger.error(f"Agent 模式处理失败: {str(e)}")
                    print("⚠️ Agent 执行异常，已自动切回普通 RAG 模式。")
                    pipeline.agent_mode_enabled = False
                    answer_dict = pipeline.query(user_input, verbose=True)
                print(answer_dict)
            else:
                if pipeline.enable_multy_turn:
                    reply, finished = pipeline.conversational_query(user_input, verbose=True)
                    if not pipeline.config.STREAM:
                        print(f"\n助手: {reply}")
                    if not finished:
                        print("（请继续补充信息）")
                    continue
                else:
                    result = pipeline.query(user_input, verbose=True)
                    if isinstance(result, dict) and "content" in result:
                        print(f"\n{result['content']}")
                        if result.get("references"):
                            print("\n--- 参考来源 ---")
                            for idx, ref in enumerate(result["references"], 1):
                                print(f"  [{idx}] {ref.get('paper_path', '未知来源')}")
                    else:
                        print(result)
        except KeyboardInterrupt:
            print("\n\n检测到中断，退出程序。")
            break
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            print("抱歉，系统遇到错误，请重试。")



def main():
    """
    主入口函数

    支持两种运行模式：
    - 无参数：交互式问答模式
    - --build：仅构建索引
    - --rebuild：强制重建索引
    """
    # Windows GBK 控制台无法编码 emoji（如 ✅/⚠️），统一使用 UTF-8 并容错，避免打印崩溃
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(description="RAG全流程构建脚本")
    parser.add_argument("--build", action="store_true", help="构建向量索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引（会清空现有数据）")
    parser.add_argument("--query", type=str, help="单次查询模式，传入问题文本")
    parser.add_argument("--config", type=str, help="自定义配置JSON文件路径（可选）")
    parser.add_argument("--add-stock", action="store_true", help="增量插入配置目录中的个股研报")
    parser.add_argument("--add-industry", action="store_true", help="增量插入配置目录中的行业研报")
    parser.add_argument("--validate-refs", type=str, help="对引用 JSON 文件执行 L1 引用核验（元素含 paper_path/text）")
    parser.add_argument("--refs-mode", type=str, default="comma", help="数字匹配口径：raw/comma/loose（默认 comma）")

    args = parser.parse_args()

    # 引用核验模式：独立运行，无需 API Key 与完整管线
    if args.validate_refs:
        _run_citation_validation(args.validate_refs, args.refs_mode)
        return

    # 初始化配置
    config = RAGConfig()

    # 检查API Key
    if not config.DASHSCOPE_API_KEY:
        print("错误: 未设置DASHSCOPE_API_KEY环境变量")
        print("请执行: export DASHSCOPE_API_KEY='your-api-key'")
        sys.exit(1)

    # 初始化Pipeline
    pipeline = RAGPipeline(config)

    if args.add_stock:
        pipeline.add_new_stock_reports()
        print("个股研报增量插入完成")
    if args.add_industry:
        pipeline.add_new_industry_reports()
        print("行业研报增量插入完成")

    if args.rebuild:
        pipeline.build_index(force_rebuild=True)
        print("索引重建完成")
    elif args.build:
        pipeline.build_index(force_rebuild=False)
        print("索引构建完成")
    elif args.query:
        answer = pipeline.query(args.query, verbose=False)
        print(answer)
    else:
        # 默认进入交互模式
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()




if __name__ == "__main__":
    main()
