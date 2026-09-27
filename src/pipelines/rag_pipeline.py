"""
pipelines/rag_pipeline.py
RAG 主流程管线 - 整合文档处理、索引构建、检索与生成
"""

import os
import time
import json
import re
import datetime
import hashlib
import pickle
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from config.rag_config import get_config
from config.rag_config import RAGConfig
from prompts.pipeline import (
    FILTER_EXTRACT_PROMPT_TEMPLATE,
    IMAGE_DETECT_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
)
from pipelines.citation_validator import CitationValidator
from agents.planner import AgentPlanner
from chains.rag_chain import LangChainRAGChain
from filters import QueryFilters
from memory import ClarifyStatus, ConversationState, MemoryStore, create_memory_store
from prompts.fallback import build_suggest_missing, log_fallback as _log_fallback
from tools.tools_registry import get_agent_tools
from core.interfaces import IGenerator, IReranker, IRetriever
from core.retrievers import BM25Retriever, HandwrittenRetriever, HybridRetriever, LangChainRetriever
from core.rerankers import RerankerAdapter, apply_file_diversity, file_keys_from_candidates
from core.generators import GeneratorAdapter


# === Missing imports from refactored modules ===
from embeddings import EmbeddingClient
from vectorstore import QdrantClientWrapper
from chains.rerank import RerankClient
from llm import LLMGenerator
from data import (
    load_markdown_documents,
    load_industry_documents,
    load_excel_metadata_by_title,
    get_best_metadata_for_title,
    split_documents,
)

logger = logging.getLogger(__name__)
def _table_html_to_text(html: str) -> str:
    """将聚合表格的 HTML 行文本转为可读纯文本。

    研报表格行以 HTML 形式存储（``<tr><td>a</td><td>b</td></tr>``），
    转为 ``a | b`` 形式的纯文本便于前端直接展示。

    Args:
        html: 聚合表格文本（可能混有非 HTML 行）

    Returns:
        单元格用 `` | `` 分隔的纯文本行；非 HTML 行原样保留。
    """
    if not html:
        return ""
    lines = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)
        if cells:
            cleaned = [re.sub(r"<[^>]+>", "", c).replace("\n", " ").strip() for c in cells]
            lines.append(" | ".join(cleaned))
    if lines:
        return "\n".join(lines)
    return html


class RAGPipeline:
    """
    RAG完整流程编排类

    整合文档处理、索引构建、检索与生成
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_client = EmbeddingClient(config)
        self.qdrant_client = QdrantClientWrapper(config)
        self.rerank_client = RerankClient(config)
        self.llm_generator = LLMGenerator(config)
        self.conversation_state = ConversationState()
        # #6 记忆持久化：按 user_id 存取会话快照（SQLite 默认 / Redis 可选；none 降级纯内存）
        self.memory_store: Optional[MemoryStore] = None
        if getattr(config, "MEMORY_BACKEND", "sqlite").lower() != "none":
            self.memory_store = create_memory_store(
                backend=config.MEMORY_BACKEND,
                db_path=config.MEMORY_DB_PATH,
                redis_url=config.MEMORY_REDIS_URL,
                ttl=config.MEMORY_TTL_SECONDS,
            )
            if self.memory_store is not None:
                logger.info("会话记忆持久化已启用（backend=%s, ttl=%ss）", config.MEMORY_BACKEND, config.MEMORY_TTL_SECONDS)
        self.agent_planner = AgentPlanner(
            llm_client=self.llm_generator.client,
            config=config,
            rag_pipeline=self,
        )
        self._langgraph_planner = None  # 实验：LangGraph 版 Agent 规划器（懒加载）
        self._langgraph_multi_planner = None  # 实验：LangGraph 多 Agent 协作规划器（懒加载）
        self.agent_mode_enabled = False  # Agent 模式开关
        self.enable_multy_turn = False # 是否启用多轮对话
        self.use_langchain_retriever = False  # 实验：LangChain 检索器开关
        self.use_hybrid_retriever = bool(getattr(config, "HYBRID_ENABLED", False))  # 实验：混合检索（向量 + BM25）开关
        self.use_langchain_chain = False    # 实验：LCEL 完整链路开关
        self.langchain_rag_chain = None
        # #3 链路收敛：query() 只依赖接口（IRetriever/IReranker/IGenerator）
        self._retriever: Optional[IRetriever] = None   # 检索器（接口），惰性构建
        self._retriever_flag: Tuple[bool, bool] = (False, False)  # 记录构建时的开关组合（langchain, hybrid）
        self._bm25_retriever: Optional[BM25Retriever] = None  # BM25 索引（#8 实验），惰性构建
        self.reranker: IReranker = RerankerAdapter(self.rerank_client)
        self.generator: IGenerator = GeneratorAdapter(self.llm_generator)
        self.citation_validator: Optional[CitationValidator] = None  # 引用核验器（L1），惰性初始化

        self._table_agg_cache: Dict[str, Dict[str, str]] = {}  # parent_id -> {text, paper_path}（查询时缓存，降低重复耗时）
        self._img_title_cache: Dict[str, str] = {}  # text-hash -> 图片标题（避免重复 LLM 调用）
        self._img_title_lock = threading.Lock()

        self._table_agg_lock = threading.Lock()
        # 路线 1：查询缓存（SQLite KV，缓解重复查询耗时）+ 会话状态并发写锁
        self.query_cache = None
        if getattr(config, "QUERY_CACHE_ENABLED", True):
            try:
                from utils.query_cache import SQLiteQueryCache
                self.query_cache = SQLiteQueryCache(
                    db_path=config.QUERY_CACHE_DB,
                    ttl=config.QUERY_CACHE_TTL,
                )
                logger.info("查询缓存已启用（db=%s, ttl=%ss）", config.QUERY_CACHE_DB, config.QUERY_CACHE_TTL)
            except Exception as exc:  # noqa: BLE001
                logger.warning("查询缓存初始化失败（降级为不缓存）: %s", exc)
                self.query_cache = None
        self._conversation_lock = threading.Lock()  # 保护 conversation_state（Agent 并行写 sql 等）
    
    def _get_agent_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_reports",
                    "description": "从研报知识库中检索信息，用于回答归因分析、公司评价、政策解读等问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "检索查询语句，应包含关键主体、具体时间和服务问题焦点"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_financial_and_visualize",
                    "description": """当问题不完整（缺少主体、时间、查询指标）时，返回反问，让用户补充。当问题完整时，根据需求生成sql查询语句，从数据库中查询结构化财务数据，并根据数据自动生成图表（如折线图、柱状图）。
                        用于回答需要财务数值、公司排名或趋势分析的问题，返回包含文字分析和图片链接的完整回答。该工具可以查询如下表格中的字段：### 1. 核心业绩指标表 (`core_performance_indicators_sheet`)
                        - `eps`: 每股收益(元)
                        - `total_operating_revenue`: 营业总收入(万元)
                        - `operating_revenue_yoy_growth`: 营业总收入-同比增长(%)
                        - `operating_revenue_qoq_growth`: 营业总收入-季度环比增长(%)
                        - `net_profit_10k_yuan`: 净利润(万元)
                        - `net_profit_yoy_growth`: 净利润-同比增长(%)
                        - `net_profit_qoq_growth`: 净利润-季度环比增长(%)
                        - `net_asset_per_share`: 每股净资产(元)
                        - `roe`: 净资产收益率(%)
                        - `operating_cf_per_share`: 每股经营现金流量(元)
                        - `net_profit_excl_non_recurring`: 扣非净利润（万元）
                        - `net_profit_excl_non_recurring_yoy`: 扣非净利润同比增长（%）
                        - `gross_profit_margin`: 销售毛利率(%)
                        - `net_profit_margin`: 销售净利率（%）
                        - `roe_weighted_excl_non_recurring`: 加权平均净资产收益率（扣非）（%）

                        ### 2. 资产负债表 (`balance_sheet`)
                        - `asset_cash_and_cash_equivalents`: 资产-货币资金(万元)
                        - `asset_accounts_receivable`: 资产-应收账款(万元)
                        - `asset_inventory`: 资产-存货(万元)
                        - `asset_trading_financial_assets`: 资产-交易性金融资产（万元）
                        - `asset_construction_in_progress`: 资产-在建工程（万元）
                        - `asset_total_assets`: 资产-总资产(万元)
                        - `asset_total_assets_yoy_growth`: 资产-总资产同比(%)
                        - `liability_accounts_payable`: 负债-应付账款(万元)
                        - `liability_advance_from_customers`: 负债-预收账款(万元)
                        - `liability_total_liabilities`: 负债-总负债(万元)
                        - `liability_total_liabilities_yoy_growth`: 负债-总负债同比(%)
                        - `liability_contract_liabilities`: 负债-合同负债（万元）
                        - `liability_short_term_loans`: 负债-短期借款（万元）
                        - `asset_liability_ratio`: 资产负债率(%)
                        - `equity_unappropriated_profit`: 股东权益-未分配利润（万元）
                        - `equity_total_equity`: 股东权益合计(万元)

                        ### 3. 现金流量表 (`cash_flow_sheet`)
                        - `net_cash_flow`: 净现金流(元) - 注意单位是元
                        - `net_cash_flow_yoy_growth`: 净现金流-同比增长(%)
                        - `operating_cf_net_amount`: 经营性现金流-现金流量净额(万元)
                        - `operating_cf_ratio_of_net_cf`: 经营性现金流-净现金流占比(%)
                        - `operating_cf_cash_from_sales`: 经营性现金流-销售商品收到的现金（万元）
                        - `investing_cf_net_amount`: 投资性现金流-现金流量净额(万元)
                        - `investing_cf_ratio_of_net_cf`: 投资性现金流-净现金流占比(%)
                        - `investing_cf_cash_for_investments`: 投资性现金流-投资支付的现金（万元）
                        - `investing_cf_cash_from_investment_recovery`: 投资性现金流-收回投资收到的现金（万元）
                        - `financing_cf_cash_from_borrowing`: 融资性现金流-取得借款收到的现金（万元）
                        - `financing_cf_cash_for_debt_repayment`: 融资性现金流-偿还债务支付的现金（万元）
                        - `financing_cf_net_amount`: 融资性现金流-现金流量净额(万元)
                        - `financing_cf_ratio_of_net_cf`: 融资性现金流-净现金流占比(%)

                        ### 4. 利润表 (`income_sheet`)
                        - `net_profit`: 净利润(万元)
                        - `net_profit_yoy_growth`: 净利润同比(%)
                        - `other_income`: 其他收益（万元）
                        - `total_operating_revenue`: 营业总收入(万元)
                        - `operating_revenue_yoy_growth`: 营业总收入同比(%)
                        - `operating_expense_cost_of_sales`: 营业总支出-营业支出(万元)
                        - `operating_expense_selling_expenses`: 营业总支出-销售费用(万元)
                        - `operating_expense_administrative_expenses`: 营业总支出-管理费用(万元)
                        - `operating_expense_financial_expenses`: 营业总支出-财务费用(万元)
                        - `operating_expense_rnd_expenses`: 营业总支出-研发费用（万元）
                        - `operating_expense_taxes_and_surcharges`: 营业总支出-税金及附加（万元）
                        - `total_operating_expenses`: 营业总支出(万元)
                        - `operating_profit`: 营业利润(万元)
                        - `total_profit`: 利润总额(万元)
                        - `asset_impairment_loss`: 资产减值损失（万元）
                        - `credit_impairment_loss`: 信用减值损失（万元）
                        """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "用户的自然语言问题，例如'云南白药2024年第三季度营收和净利润是多少'或'近三年片仔癀利润趋势'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def get_langchain_retriever(self):
        """
        使用 LangChain QdrantVectorStore 构建检索器。

        通过 self.get_vectorstore() 获取带适配器的向量存储，
        返回 as_retriever() 以便在 query() 中替代原有 search_similar 逻辑。
        """
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config.RETRIEVAL_K}
        )
        logger.info("LangChain 检索器已初始化 (k=%d)", self.config.RETRIEVAL_K)
        return retriever

    def get_vectorstore(self):
        """
        获取 LangChain 的 QdrantVectorStore 实例（使用 get_vector_store_direct，跳过维度验证）。

        通过 config.rag_config.get_config().get_vector_store_direct() 获取，
        返回 QdrantVectorStore 以供 similarity_search_with_score 使用。
        """
        if not hasattr(self, '_vectorstore'):
            cfg = get_config()
            embedding = cfg.get_embedding_adapter()  # 使用适配器
            client = self.qdrant_client.client  # QdrantClientWrapper 中的 self.client
            self._vectorstore = cfg.get_vector_store_direct(
                client=client,
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                embedding=embedding,
            )
            logger.info("LangChain 向量存储已初始化 (collection=%s)", self.config.QDRANT_COLLECTION_NAME)
        return self._vectorstore

    def _get_retriever(self) -> IRetriever:
        """返回当前检索器（IRetriever 接口）。
        默认手写实现（HandwrittenRetriever）；use_langchain_retriever / use_hybrid_retriever 为实验开关，
        打开时分别切换到 LangChainRetriever / 混合检索（向量 + BM25），开关变化后自动重建。
        """
        flag = (self.use_langchain_retriever, self.use_hybrid_retriever)
        if self._retriever is None or self._retriever_flag != flag:
            if self.use_hybrid_retriever:
                logger.info("实验链路：混合检索（向量 + BM25，RRF 融合）")
                self._retriever = HybridRetriever(
                    vector_retriever=HandwrittenRetriever(
                        embedding_client=self.embedding_client,
                        qdrant_client=self.qdrant_client,
                        top_k=self.config.HYBRID_TOPK_VECTOR,
                    ),
                    bm25_retriever=self.build_bm25_index(),
                    top_k=self.config.RETRIEVAL_K,
                    rrf_k=self.config.HYBRID_RRF_K,
                    topk_vector=self.config.HYBRID_TOPK_VECTOR,
                    topk_bm25=self.config.HYBRID_TOPK_BM25,
                    vector_floor_ratio=self.config.HYBRID_VECTOR_FLOOR_RATIO,
                )
            elif flag[0]:
                logger.info("实验链路：LangChain 检索器")
                self._retriever = LangChainRetriever(
                    vectorstore=self.get_vectorstore(),
                    top_k=self.config.RETRIEVAL_K,
                )
            else:
                self._retriever = HandwrittenRetriever(
                    embedding_client=self.embedding_client,
                    qdrant_client=self.qdrant_client,
                    top_k=self.config.RETRIEVAL_K,
                )
            self._retriever_flag = flag
        return self._retriever

    def build_bm25_index(self, force: bool = False) -> BM25Retriever:
        """从 Qdrant 全量点构建 BM25 索引（#8 混合检索实验）。

        首次调用时滚动拉取集合全部点构建并缓存；索引文件按
        "集合名+点数" 命名落盘，增量插入后点数变化会自动重建；
        force=True 时强制重建。

        Returns:
            BM25Retriever 实例（内部缓存，重复调用返回同一实例）
        """
        if self._bm25_retriever is not None and not force:
            return self._bm25_retriever
        count = self.qdrant_client.count()
        index_path = Path(self.config.BM25_INDEX_PATH)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        target = index_path.with_name(
            f"{index_path.stem}.{self.config.QDRANT_COLLECTION_NAME}.{count}.pkl"
        )
        if not force and target.exists():
            with open(target, "rb") as fh:
                self._bm25_retriever = pickle.load(fh)
            logger.info("BM25 索引从缓存加载：%s（%d 个文档块）", target.name, count)
            return self._bm25_retriever
        docs = self.qdrant_client.scroll_all()
        if not docs:
            raise RuntimeError("Qdrant 集合为空，无法构建 BM25 索引，请先 build_index")
        self._bm25_retriever = BM25Retriever(docs=docs)
        with open(target, "wb") as fh:
            pickle.dump(self._bm25_retriever, fh)
        logger.info("BM25 索引构建完成：%d 个文档块 → %s", len(docs), target.name)
        return self._bm25_retriever

    def validate_citations(self, references: List[Dict[str, Any]], match_mode: Optional[str] = None) -> Dict[str, Any]:
        """对引用列表执行 L1 核验（文件可溯源 + 数字可溯源）。

        Args:
            references: 引用列表，元素须含 paper_path / text 字段
                        （如 {paper_path, text, paper_image}）
            match_mode: 数字匹配口径（raw / comma / loose），默认取配置 CITATION_MATCH_MODE

        Returns:
            包含逐条核验记录与汇总统计的字典：{"records": [...], "summary": {...}}
        """
        if self.citation_validator is None:
            self.citation_validator = CitationValidator(
                corpus_root=self.config.CITATION_CORPUS_ROOT,
                match_mode=match_mode or self.config.CITATION_MATCH_MODE,
            )
        elif match_mode is not None:
            self.citation_validator.match_mode = match_mode
        records = self.citation_validator.check_references(references)
        summary = self.citation_validator.summarize(records)
        return {"records": records, "summary": summary}

    def _filter_citations(self, references: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
        """对引用执行 L1 核验并按配置过滤。

        用于在线查询返回前拦截文件缺失等高风险引用（如 B2049 / B2063 类模型生成式引用）。

        Args:
            references: 引用列表（{paper_path, text, paper_image}）
            verbose: 是否打印核验日志

        Returns:
            过滤后的引用列表
        """
        if not references or not self.config.CITATION_CHECK_ON_QUERY:
            return references
        result = self.validate_citations(references)
        records = result["records"]
        summary = result["summary"]
        dropped = []
        kept = []
        for ref, rec in zip(references, records):
            if self.config.CITATION_FILTER_MISSING and rec["status"] == "missing":
                dropped.append(("missing", ref.get("paper_path", "")))
                continue
            if (
                self.config.CITATION_FILTER_ZERO_HIT
                and rec["status"] != "missing"
                and rec["nums"] > 0
                and rec["num_hit"] == 0
            ):
                dropped.append(("zero_hit", ref.get("paper_path", "")))
                continue
            # 附带核验状态，供前端展示引用可信度
            ref["citation"] = {
                "status": rec["status"],
                "located": rec["located"],
                "nums": rec["nums"],
                "num_hit": rec["num_hit"],
                "num_ratio": rec["num_ratio"],
                "unhit": rec["unhit"],
                "hits_context": rec.get("hits_context") or [],
            }
            kept.append(ref)
        if dropped and verbose:
            print(f"[引用核验] {len(references)} 条引用，过滤 {len(dropped)} 条：{dropped}")
        elif verbose and summary.get("total"):
            print(f"[引用核验] {len(references)} 条引用全部通过（文件可溯源 {summary['traceable']}/{summary['total']}）")
        return kept

    def _generate_clarify_question(self, missing_fields: List[str]) -> str:
        """根据缺失字段生成自然语言提问"""
        field_prompts = {
            "stock_name": "请问您想查询哪家公司的信息？",
            "start_date": "请问您想查询哪个时间段的研报？（例如：2024年、近三个月）",
        }
        questions = [field_prompts.get(f, f"请提供 {f}") for f in missing_fields]
        return "；".join(questions)

    def _parse_filters_with_llm(self, question: str, history: List[Dict] = None) -> QueryFilters:
        """使用 LLM 从问题中提取过滤条件"""
        # 构造历史对话文本
        history_text = ""
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])  # 最近4轮
        # 获取并格式化当前日期，作为LLM的计算基准
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        # 获取当前年份，辅助LLM理解“今年”、“明年”等概念
        current_year = datetime.datetime.now().year
        history_section = ("对话历史：" + history_text) if history_text else ""
        prompt = FILTER_EXTRACT_PROMPT_TEMPLATE.format(
            history_section=history_section,
            today=today,
            current_year=current_year,
            question=question,
        )

        try:
            # 使用已有的 LLM 客户端进行单次调用（非流式）
            response = self.llm_generator.client.chat.completions.create(
                model="qwen-turbo",  # 轻量模型，降低成本
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256
            )
            content = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块标记
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return QueryFilters(**data)
        except Exception as e:
            logger.warning(f"LLM 解析过滤条件失败，将不使用过滤: {e}")
            return QueryFilters()
    
    def _should_aggregate_table(self, query: str) -> bool:
        """根据查询关键词判断是否需要将表格行合并为完整表格"""
        keywords = ["趋势", "变化", "对比", "历年", "过去", "逐年", "增长趋势", "下降趋势", "近三年", "近五年"]
        query_lower = query.lower()
        return True #if any(kw in query_lower for kw in keywords) else False

    def _aggregate_parent_table(
        self,
        search_results: List[Dict[str, Any]],
        candidate_docs: List[str],
        max_table_len: int = 25000
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        聚合检索结果中的表格行：命中任意一行，则拉取整个表格。
        返回：新的文档内容列表、新的元数据列表（用于后续可能追溯）
        """
        table_metadata = {}  # key: 聚合表格在 new_candidate_docs 中的索引，value: 元数据字典
        # 收集需要聚合的 parent_id（仅限排序后 top-K 候选，低相关表格行保持原样，避免首查多表 scroll 拖慢）
        top_k = getattr(self.config, "TABLE_AGG_TOPK", 20)
        scope = search_results[:top_k] if top_k and 0 < top_k < len(search_results) else search_results
        parent_ids = set()
        row_indices_map = {}  # parent_id -> 列表，记录在 search_results 中出现的位置
        for i, res in enumerate(scope):
            payload = res["payload"]
            if payload.get("is_table_row") and payload.get("parent_id"):
                pid = payload["parent_id"]
                parent_ids.add(pid)
                if pid not in row_indices_map:
                    row_indices_map[pid] = []
                row_indices_map[pid].append(i)

        if not parent_ids:
            return candidate_docs, search_results, table_metadata, {}  # 无表格行，直接返回原样（空 index_map）

        # 从 Qdrant 中按 parent_id 拉取所有子块（内存缓存 + 并发 scroll，降低重复/多父表耗时）
        aggregated_tables = {}
        with self._table_agg_lock:
            cached = {pid: val for pid, val in self._table_agg_cache.items() if pid in parent_ids}
        miss_ids = [pid for pid in parent_ids if pid not in cached]

        def _scroll_table(pid: int) -> Optional[Dict[str, str]]:
            points, _ = self.qdrant_client.client.scroll(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                scroll_filter=self.qdrant_client.models.Filter(
                    must=[
                        self.qdrant_client.models.FieldCondition(
                            key="parent_id",
                            match=self.qdrant_client.models.MatchValue(value=pid)
                        )
                    ]
                ),
                limit=200,  # 假设一个表格最多200行
                with_payload=True,
                with_vectors=False
            )
            if not points:
                return None
            # 按 row_index 排序
            sorted_points = sorted(points, key=lambda p: p.payload.get("row_index", 0))
            table_text = "\n".join([p.payload["content"] for p in sorted_points])
            if len(table_text) > max_table_len:
                table_text = table_text[:max_table_len] + "\n...(表格内容过长，已截断)"
            sources = list(set(p.payload.get("source", "") for p in points))
            return {
                "text": table_text,
                "paper_path": "; ".join(sources) if sources else "聚合表格/多源",
            }

        if miss_ids:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for pid, table in zip(miss_ids, executor.map(_scroll_table, miss_ids)):
                    if table is None:
                        continue
                    aggregated_tables[pid] = table["text"]
                    table_metadata[pid] = {"paper_path": table["paper_path"]}
            with self._table_agg_lock:
                self._table_agg_cache.update(
                    {pid: {"text": aggregated_tables[pid], "paper_path": table_metadata[pid]["paper_path"]} for pid in aggregated_tables}
                )
                if len(self._table_agg_cache) > 5000:
                    self._table_agg_cache.clear()
        for pid, val in cached.items():
            aggregated_tables[pid] = val["text"]
            table_metadata[pid] = {"paper_path": val["paper_path"]}

        # 构建新的候选文档列表：用聚合后的表格替换原有的零散行。
        # index_map 记录 新候选位置 -> 原始 search_results 索引，供引用构建/文件多样性对齐（聚合后两列表索引不再一一对应）
        new_candidate_docs = []
        used_parents = set()
        aggregated_meta = {}
        index_map: Dict[int, int] = {}
        for i, doc in enumerate(candidate_docs):
            payload = search_results[i]["payload"]
            pid = payload.get("parent_id")
            if pid and pid in aggregated_tables:
                if pid not in used_parents:
                    new_idx = len(new_candidate_docs)
                    new_candidate_docs.append(aggregated_tables[pid])
                    aggregated_meta[new_idx] = table_metadata[pid]  # 保存元数据
                    index_map[new_idx] = i
                    used_parents.add(pid)
                # 已添加过聚合表格，跳过当前行
            else:
                new_idx = len(new_candidate_docs)
                new_candidate_docs.append(doc)
                index_map[new_idx] = i

        return new_candidate_docs, search_results, aggregated_meta, index_map
    
    def _generate_summaries(self, texts: List[str], batch_size: int = 20) -> List[str]:
        """
        批量生成文本摘要。对于空文本或过短文本，直接返回原文前50字。
        使用 qwen-turbo 模型以降低成本。
        """
        summaries = []
        # 分批调用 API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_summaries = []
            for text in batch:
                if not text or len(text) < 100:
                    # 文本过短，直接截取
                    batch_summaries.append(text[:100] + ("..." if len(text) > 100 else ""))
                    continue

                prompt = SUMMARY_PROMPT_TEMPLATE.format(text=text)

                try:
                    response = self.llm_generator.client.chat.completions.create(
                        model="qwen-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=80
                    )
                    summary = response.choices[0].message.content.strip()
                    batch_summaries.append(summary)
                except Exception as e:
                    logger.warning(f"摘要生成失败，回退截断: {e}")
                    batch_summaries.append(text[:100] + "...")
            summaries.extend(batch_summaries)
            # 避免请求过快
            time.sleep(0.5)
        return summaries

    def build_index(self, force_rebuild: bool = False):
        """
        构建向量索引

        步骤：加载文档 → 分块 → 生成Embedding → 存入Qdrant

        Args:
            force_rebuild: 是否强制重建（清空已有数据）
        """
        if force_rebuild:
            logger.warning("强制重建索引，将清空现有数据")
            self.qdrant_client.clear_collection()

        # 1. 加载个股研报
        stock_docs = load_markdown_documents(
            self.config.MARKDOWN_DIR,
            excel_metadata_path=self.config.EXCEL_METADATA_PATH
        )
        # 加载行业研报
        industry_docs = load_industry_documents(
            self.config.INDUSTRY_MARKDOWN_DIR,
            excel_metadata_path=self.config.INDUSTRY_EXCEL_PATH
        )
        all_docs = stock_docs + industry_docs
        logger.info(f"总计加载 {len(all_docs)} 篇研报（个股 {len(stock_docs)}，行业 {len(industry_docs)}）")
        if not all_docs:
            logger.error("未找到任何Markdown文档，请检查路径")
            return

        # 2. 分块
        chunks = split_documents(
            all_docs,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        # ----- 为非表格行生成摘要 -----
        non_table_texts = []
        non_table_indices = []
        for idx, chunk in enumerate(chunks):
            if not chunk["metadata"].get("is_table_row", False):
                non_table_texts.append(chunk["content"])
                non_table_indices.append(idx)

        if non_table_texts:
            logger.info(f"开始为 {len(non_table_texts)} 个文本块生成摘要...")
            summaries = self._generate_summaries(non_table_texts)
            for idx, summary in zip(non_table_indices, summaries):
                chunks[idx]["metadata"]["summary"] = summary
            logger.info("摘要生成完成。")

        # 3. 生成Embedding
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_client.generate_embeddings(texts, text_type="document")

        # 4. 构建payload（包含文本和元数据）
        payloads = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "content": chunk["content"],
                **chunk["metadata"]
            }
            payloads.append(payload)

        # 5. 存入Qdrant
        self.qdrant_client.insert_vectors(embeddings, payloads)

        logger.info("索引构建完成！")
    
    def add_new_documents(self, new_doc_contents: List[Dict[str, str]],
                          excel_metadata_path: Optional[str] = None) -> None:
        """
        将一批新文档增量插入向量数据库。
        new_doc_contents 结构: [{"content": "...", "metadata": {...}}, ...]
        """
        if not new_doc_contents:
            logger.info("没有需要插入的新文档")
            return

        # 1. 元数据匹配（如果提供了Excel）
        if excel_metadata_path:
            metadata_dict = load_excel_metadata_by_title(excel_metadata_path)
            for doc in new_doc_contents:
                title = doc["metadata"].get("title", "")
                matched = get_best_metadata_for_title(title, metadata_dict)
                if matched:
                    doc["metadata"].update(matched)

        # 2. 分块
        chunks = split_documents(
            new_doc_contents,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

        if not chunks:
            logger.info("分块后无有效数据，跳过插入")
            return

        # 3. 为非表格块生成摘要（与 build_index 行为一致）
        non_table_texts, non_table_indices = [], []
        for idx, chunk in enumerate(chunks):
            if not chunk["metadata"].get("is_table_row", False):
                non_table_texts.append(chunk["content"])
                non_table_indices.append(idx)
        if non_table_texts:
            summaries = self._generate_summaries(non_table_texts)
            for idx, summary in zip(non_table_indices, summaries):
                chunks[idx]["metadata"]["summary"] = summary

        # 4. Embedding
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_client.generate_embeddings(texts, text_type="document")

        # 5. 构建payload并插入（自动生成不重复ID）
        payloads = []
        for chunk in chunks:
            payload = {"content": chunk["content"], **chunk["metadata"]}
            payloads.append(payload)

        self.qdrant_client.insert_vectors(embeddings, payloads)
        logger.info(f"增量插入完成，新增 {len(embeddings)} 个向量")

    def add_new_stock_reports(self, directory: Optional[str] = None) -> None:
        """
        增量插入个股研报。默认使用 config.NEW_STOCK_DIR，
        也可临时传入 directory 覆盖。
        """
        target_dir = directory or self.config.NEW_STOCK_DIR
        if not target_dir:
            logger.warning("未指定个股研报目录 (NEW_STOCK_DIR 为空)，跳过增量插入")
            return
        if not Path(target_dir).exists():
            logger.error(f"个股研报目录不存在: {target_dir}")
            return
        logger.info(f"开始增量插入个股研报，目录: {target_dir}")
        new_docs = load_markdown_documents(target_dir, self.config.EXCEL_METADATA_PATH)
        self.add_new_documents(new_docs, self.config.EXCEL_METADATA_PATH)

    def add_new_industry_reports(self, directory: Optional[str] = None) -> None:
        """
        增量插入行业研报。默认使用 config.NEW_INDUSTRY_DIR，
        也可临时传入 directory 覆盖。
        """
        target_dir = directory or self.config.NEW_INDUSTRY_DIR
        if not target_dir:
            logger.warning("未指定行业研报目录 (NEW_INDUSTRY_DIR 为空)，跳过增量插入")
            return
        if not Path(target_dir).exists():
            logger.error(f"行业研报目录不存在: {target_dir}")
            return
        logger.info(f"开始增量插入行业研报，目录: {target_dir}")
        new_docs = load_industry_documents(target_dir, self.config.INDUSTRY_EXCEL_PATH)
        self.add_new_documents(new_docs, self.config.INDUSTRY_EXCEL_PATH)
    
    def _extract_image_title_with_llm(self, text: str) -> str:
        """
        使用 LLM 判断文本块中是否包含图片，并返回图片的标题/说明。
        如果没有图片或无法提取，返回空字符串。
        """
        # 快速检查：文本中是否包含 ![]( 标记
        if '![]' not in text:
            return ""

        cache_key = hashlib.md5(text[:4000].encode("utf-8")).hexdigest()
        with self._img_title_lock:
            if cache_key in self._img_title_cache:
                return self._img_title_cache[cache_key]

        prompt = IMAGE_DETECT_PROMPT_TEMPLATE.format(text=text[:4000])
        try:
            response = self.llm_generator.client.chat.completions.create(
                model="qwen3-max",  # 轻量模型，速度快
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128
            )
            title = response.choices[0].message.content.strip()
            if title == "无图片" or not title:
                return ""
            with self._img_title_lock:
                self._img_title_cache[cache_key] = title
            return title
        except Exception as e:
            logger.warning(f"LLM提取图片标题失败: {e}")
            return ""

    def _build_reference_for_doc(self, idx: int, search_results: List[Dict], candidate_docs: List[str], aggregated_meta: Dict[int, Dict[str, Any]] = None, index_map: Optional[Dict[int, int]] = None) -> Dict[str, str]:
        """根据候选文档索引构建引用条目，提取图表标题作为 paper_image。

        idx 是聚合后候选列表的位置；index_map（聚合返回）负责回映射到原始 search_results 索引，
        避免聚合导致 paper_path/摘要取错文件。
        """
        orig_idx = index_map.get(idx, idx) if index_map else idx
        if aggregated_meta and idx in aggregated_meta:
            meta = aggregated_meta[idx]
            paper_path = meta.get("paper_path", "聚合表格/多源")
            full_text = candidate_docs[idx]
            # 聚合表格：转纯文本作为引用内容，避免"查看原文"只有占位文案
            summary_text = _table_html_to_text(full_text)
            if len(summary_text) > 1200:
                summary_text = summary_text[:1200] + "\n...(表格内容过长，已截断)" 
        elif orig_idx < len(search_results):
            payload = search_results[orig_idx]["payload"]
            paper_path = payload.get("source", "")
            full_text = candidate_docs[idx] if idx < len(candidate_docs) else payload.get("content", "")
            # 优先使用预生成的摘要
            summary_text = payload.get("summary")
            if not summary_text:
                summary_text = full_text[:200]
        else:
            paper_path = "未知来源"
            full_text = candidate_docs[idx] if idx < len(candidate_docs) else ""
            summary_text = full_text[:200]

        paper_image = ""
        chart_pattern = r'图表\s*\d+\s*[：:]\s*[^\n]+'
        match = re.search(chart_pattern, full_text)
        if match:
            paper_image = match.group(0).strip()

        # 如果正则没有匹配到，尝试用 LLM 提取
        if not paper_image:
            paper_image = self._extract_image_title_with_llm(full_text)

        # 表格占位符（研报 md 中的 [TABLE_PLACEHOLDER_N] / [表格_N]）在引文中无内容，
        # 替换为明确提示，避免"查看原文"只看到占位符（表格行引用已由聚合逻辑补充内容）
        summary_text = re.sub(
            r"\[TABLE_PLACEHOLDER_\d+\]|\[表格_\d+\]",
            "[表格数据详见研报原文，此处省略]",
            summary_text,
        )

        return {
            "paper_path": paper_path,
            "text": summary_text,
            "paper_image": paper_image
        }
    
    def _load_conversation(self, user_id: str) -> ConversationState:
        """按 user_id 加载会话状态（优先持久化存储；缺失/过期回退新会话）"""
        if self.memory_store is not None:
            try:
                data = self.memory_store.get(user_id)
                if data:
                    state = ConversationState.from_dict(data)
                    if state.user_id == user_id:
                        return state
            except Exception as e:  # noqa: BLE001
                logger.warning("会话记忆加载失败(user_id=%s): %s", user_id, e)
        return ConversationState(user_id=user_id)

    def _save_conversation(self, state: ConversationState) -> None:
        """持久化会话状态快照（按 user_id）"""
        if self.memory_store is None:
            return
        try:
            self.memory_store.set(state.user_id, state.to_dict(), ttl=self.config.MEMORY_TTL_SECONDS)
        except Exception as e:  # noqa: BLE001
            logger.warning("会话记忆保存失败(user_id=%s): %s", state.user_id, e)

    def reset_conversation(self, user_id: str = "default") -> None:
        """清空指定用户的会话记忆（内存 + 持久化）"""
        if self.memory_store is not None:
            try:
                self.memory_store.delete(user_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("会话记忆删除失败(user_id=%s): %s", user_id, e)
        if self.conversation_state.user_id == user_id:
            self.conversation_state = ConversationState(user_id=user_id)

    def _get_agent_planner(self) -> Any:
        """按配置返回 Agent 编排后端：handwritten=自研循环（默认）/ langgraph=StateGraph 版 / langgraph+multi-agent=多 Agent 协作（实验）。"""
        if getattr(self.config, "AGENT_PLANNER_BACKEND", "handwritten") == "langgraph":
            if getattr(self.config, "AGENT_LANGGRAPH_MULTI_AGENT", False):
                if self._langgraph_multi_planner is None:
                    from agents.langgraph_multi_agent import LangGraphMultiAgentPlanner

                    self._langgraph_multi_planner = LangGraphMultiAgentPlanner(
                        llm_client=self.llm_generator.client,
                        config=self.config,
                        rag_pipeline=self,
                    )
                return self._langgraph_multi_planner
            if self._langgraph_planner is None:
                from agents.langgraph_planner import LangGraphPlanner

                self._langgraph_planner = LangGraphPlanner(
                    llm_client=self.llm_generator.client,
                    config=self.config,
                    rag_pipeline=self,
                )
                logger.info("Agent 编排后端已切换为 LangGraph（实验）")
            return self._langgraph_planner
        return self.agent_planner

    def agent_query(self, question: str, user_id: str = "default", verbose: bool = True, on_stage: Optional[Callable[[str], None]] = None, on_chunk: Optional[Callable[[str], None]] = None) -> dict:
        # B-28 错字归一化（Agent 入口统一处理，覆盖 handwritten / langgraph 两后端）：
        # 在拆解前还原公司简称/指标错字（云白要→云南白药、资产负债绿→资产负债率 等），
        # 避免 supervisor 先改写错字导致后续无法纠回（C2011 根因）
        if question and question.strip():
            try:
                from tools.typo_normalizer import load_company_abbrs_from_config, normalize_question_typos  # noqa: PLC0415

                names = load_company_abbrs_from_config(self.config)
                norm_question, fixes = normalize_question_typos(question, names)
                if fixes:
                    logger.info("B-28 错字归一化(agent_query 入口): %r -> %r（%s）", question, norm_question, fixes)
                question = norm_question
            except Exception as exc:  # noqa: BLE001
                logger.warning("B-28 错字归一化跳过（不影响主流程）: %s", exc)
        # 按 user_id 加载会话（持久化存储优先；重启后跨进程恢复）
        state = self._load_conversation(user_id)
        self.conversation_state = state
        # 检查超时
        current_time = time.time()
        if current_time - state.last_active > self.config.CONVERSATION_TIMEOUT_SECONDS:
            state = ConversationState(user_id=user_id)
            self.conversation_state = state
            if verbose:
                print("[系统] 对话已超时，已开始新话题。")
        state.last_active = current_time

        # 意图清晰，执行真正的 Agent，并传入历史（后端：handwritten 自研默认 / langgraph 实验）
        answer = self._get_agent_planner().execute(
            user_query=question,
            history=state.history,
            user_id=user_id,
            verbose=verbose,
            on_stage=on_stage,
            on_chunk=on_chunk,
        )
        # answer 现在是字典
        state.history.append({"role": "user", "content": question})
        state.history.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
        state.last_active = time.time()
        state.pending_question = None
        state.status = ClarifyStatus.READY

        # 将本轮回答加入 rounds 列表
        state.rounds.append(answer)

        # 持久化会话快照（按 user_id）
        self._save_conversation(state)

        return answer

    def get_accumulated_sql(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """返回指定用户会话的所有sql语句"""
        if self.conversation_state.user_id == user_id:
            return self.conversation_state.sql
        # 当前会话不是该用户时，尝试从持久化存储恢复
        if self.memory_store is not None:
            try:
                data = self.memory_store.get(user_id)
                if data:
                    return ConversationState.from_dict(data).sql
            except Exception:  # noqa: BLE001
                pass
        return []
    
    def conversational_query(self, user_input: str, user_id: str = "default", verbose: bool = True) -> Tuple[str, bool]:
        """
        执行一轮对话查询。
        返回: (回复内容, 是否已完成一次完整问答)
        """
        # B-28 错字归一化（Agent 入口统一处理，覆盖 handwritten / langgraph 两后端）：
        # 在拆解前还原公司简称/指标错字（云白要→云南白药、资产负债绿→资产负债率 等），
        # 避免 supervisor 先改写错字导致后续无法纠回（C2011 根因）
        if user_input and user_input.strip():
            try:
                from tools.typo_normalizer import load_company_abbrs_from_config, normalize_question_typos  # noqa: PLC0415

                names = load_company_abbrs_from_config(self.config)
                normalized_input, fixes = normalize_question_typos(user_input, names)
                if fixes:
                    logger.info("B-28 错字归一化(conversational_query 入口): %r -> %r（%s）", user_input, normalized_input, fixes)
                user_input = normalized_input
            except Exception as exc:  # noqa: BLE001
                logger.warning("B-28 错字归一化跳过（不影响主流程）: %s", exc)
        # 按 user_id 加载会话（持久化存储优先；重启后跨进程恢复）
        state = self._load_conversation(user_id)
        self.conversation_state = state
        current_time = time.time()
        if current_time - state.last_active > self.config.CONVERSATION_TIMEOUT_SECONDS:  # 5分钟超时
            state = ConversationState(user_id=user_id)
            self.conversation_state = state
            if verbose:
                print("[系统] 对话已超时，已开始新话题。")
        state.last_active = current_time
        # 1. 更新历史
        self.conversation_state.history.append({"role": "user", "content": user_input})

        # 2. 解析过滤条件（传入历史，使LLM能理解代词指代）
        new_filters = self._parse_filters_with_llm(
            user_input,
            self.conversation_state.history[:-1]  # 除本轮外的历史
        )
        # 合并非空字段到状态中的filters
        for field in new_filters.__dataclass_fields__:
            value = getattr(new_filters, field)
            if value is not None:
                setattr(self.conversation_state.filters, field, value)

        # 3. 检查关键字段缺失
        missing = []
        if not self.conversation_state.filters.stock_name:
            missing.append("stock_name")
        # 可扩展：检查其他必填字段（如日期有时也关键）

        if missing:
            question = self._generate_clarify_question(missing)
            self.conversation_state.status = ClarifyStatus.NEED_CLARIFY
            self.conversation_state.clarify_question = question
            reply, template_id = build_suggest_missing(list(missing), clarify_question=question)
            _log_fallback(template_id, detail="clarify_missing_" + ",".join(missing))
            self.conversation_state.history.append({"role": "assistant", "content": reply})
            self._save_conversation(self.conversation_state)
            return reply, False  # 未结束，等待用户补充

        # 4. 信息充分，执行检索
        self.conversation_state.status = ClarifyStatus.READY
        if verbose:
            print(f"[过滤条件] {self.conversation_state.filters}")

        # 向量检索（软过滤：不硬过滤，召回后用元数据匹配得分重排，与 query() 保持一致） 
        search_results = self._get_retriever().retrieve(user_input, query_filter=None)
        # 软约束：融合元数据匹配得分，重新排序
        alpha = 0.8
        for res in search_results:
            vector_score = res["score"]
            match_score = self.conversation_state.filters.compute_match_score(res["payload"])
            res["combined_score"] = alpha * vector_score + (1 - alpha) * match_score
        search_results.sort(key=lambda x: x["combined_score"], reverse=True)
        search_results = search_results[:self.config.RETRIEVAL_K]

        if verbose:
            print(f"\n[检索阶段] 召回 {len(search_results)} 个相关片段")

        candidate_docs = [res["payload"]["content"] for res in search_results]

        # 可选：父表聚合
        if self._should_aggregate_table(user_input):
            if verbose:
                print("[聚合模式] 检测到趋势/对比类问题，将合并相关表格行")
            candidate_docs, _, aggregated_meta, agg_index_map = self._aggregate_parent_table(search_results, candidate_docs)
        else:
            aggregated_meta = {}
            agg_index_map = {}

        # Rerank（请求 oversample 条，再按每文件上限收敛回 RERANK_TOP_N）
        rerank_results = self.reranker.rerank(
            query=user_input,
            documents=candidate_docs,
            top_n=self.config.RERANK_TOP_N * self.config.RERANK_OVERSAMPLE
        )
        # 多样性按候选列表对齐文件键（聚合后 candidate_docs 与 search_results 索引不再一一对应）
        diversity_keys = (
            [search_results[agg_index_map[i]] for i in range(len(candidate_docs))] if agg_index_map else search_results
        )
        rerank_results = apply_file_diversity(
            rerank_results,
            file_keys_from_candidates(diversity_keys),
            self.config.RERANK_TOP_N,
            self.config.RERANK_MAX_PER_FILE,
        )

        final_contexts = []
        for item in rerank_results:
            doc_text = item.get("document")
            if not doc_text:
                idx = item["index"]
                doc_text = candidate_docs[idx] if idx < len(candidate_docs) else ""
            final_contexts.append(doc_text)

        if verbose:
            print(f"[重排序阶段] 保留前 {len(rerank_results)} 个最相关片段\n")
            for i, item in enumerate(rerank_results, 1):
                print(f"  [{i}] 相关性分数: {item['relevance_score']:.4f}")

        # 生成回答
        if verbose:
            print("\n[生成回答]\n")

        # 注意：这里假设 generate 方法可以接收历史对话，若不能，先仅使用当前上下文
        answer = self.generator.generate(
            query=user_input,
            contexts=final_contexts,
            history=self.conversation_state.history[:-1],  # 传入本轮之前的历史
            stream=self.config.STREAM
        )

        self.conversation_state.history.append({"role": "assistant", "content": answer})
        self.conversation_state.status = ClarifyStatus.READY
        self.conversation_state.last_active = time.time()  # 回答完毕后也更新活跃时间
        self._save_conversation(self.conversation_state)
        return answer, True
    
    def query(self, question: str, verbose: bool = True, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """RAG 问答入口。
        默认手写链路（由 IRetriever/IReranker/IGenerator 接口编排）；
        use_langchain_chain / use_langchain_retriever 为实验开关，仅调试用。
        """
        # 0. 【实验】LCEL 完整链路分支（跳过手写 Embedding/检索/Rerank/生成逻辑）
        if self.use_langchain_chain:
            print("[QUERY DEBUG] 进入 LCEL 分支")
            try:
                if self.langchain_rag_chain is None:
                    from chains.rag_chain import LangChainRAGChain
                    retriever = self.get_langchain_retriever()
                    self.langchain_rag_chain = LangChainRAGChain(retriever=retriever)
                result = self.langchain_rag_chain.invoke(question)
                print(f"[QUERY DEBUG] invoke 返回结果: {repr(result)}")
                return {"content": result, "image": [], "references": []}
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"LCEL 链路执行异常: {e}")
                return {"content": f"LCEL 链路执行异常: {e}", "image": [], "references": []}
        # 1. 检索（默认手写链路；use_langchain_retriever 为实验链路） 
        filter_box: Dict[str, Any] = {}
        filter_thread = threading.Thread(
            target=lambda: filter_box.__setitem__("filters", self._parse_filters_with_llm(question)),
            daemon=True,
        )
        filter_thread.start()
        query_filter = None  # 硬过滤已架空，仅保留软约束

        search_results = self._get_retriever().retrieve(question, query_filter=query_filter)
        if not search_results and (self.use_langchain_retriever or self.use_hybrid_retriever):
            # 实验链路：空召回直接返回（与历史行为一致） 
            return {"content": "未找到相关内容，请尝试其他问题。", "image": [], "references": []}
        # 过滤条件解析与检索并行：检索完成后等待其结束（超时则放弃软过滤）
        filter_thread.join(timeout=8.0)
        filters = filter_box.get("filters") or QueryFilters()
        if verbose and filters.has_any_filter():
            print(f"[过滤条件（软约束）] {filters}")
        if verbose:
            if self.use_hybrid_retriever:
                label = "（混合检索：向量 + BM25）"
            elif self.use_langchain_retriever:
                label = "（LangChain 检索器）"
            else:
                label = ""
            print(f"\n[检索阶段] 召回 {len(search_results)} 个相关片段{label}")
        # 4. 软约束：融合元数据匹配得分，重新排序
        alpha = 0.8  # 向量相似度权重，匹配度权重为 1-alpha
        for res in search_results:
            vector_score = res["score"]  # Qdrant 返回的相似度（余弦距离转换后通常在 0~1）
            match_score = filters.compute_match_score(res["payload"])
            res["combined_score"] = alpha * vector_score + (1 - alpha) * match_score

        # 按组合得分降序排序
        search_results.sort(key=lambda x: x["combined_score"], reverse=True)
        # 取前 RETRIEVAL_K 个进入后续处理
        search_results = search_results[:self.config.RETRIEVAL_K]

        # 提取文档内容
        candidate_docs = [res["payload"]["content"] for res in search_results]

        # ----- 判断是否需要父表聚合 -----
        if self._should_aggregate_table(question):
            if verbose:
                print("[聚合模式] 检测到趋势/对比类问题，将合并相关表格行")
            candidate_docs, _, aggregated_meta, agg_index_map = self._aggregate_parent_table(search_results, candidate_docs)
        else:
            aggregated_meta = {}
            agg_index_map = {}

        # 5. Rerank 重排序（请求 oversample 条，再按每文件上限收敛回 RERANK_TOP_N）
        rerank_results = self.reranker.rerank(
            query=question,
            documents=candidate_docs,
            top_n=self.config.RERANK_TOP_N * self.config.RERANK_OVERSAMPLE
        )
        # 多样性按候选列表对齐文件键（聚合后 candidate_docs 与 search_results 索引不再一一对应）
        diversity_keys = (
            [search_results[agg_index_map[i]] for i in range(len(candidate_docs))] if agg_index_map else search_results
        )
        rerank_results = apply_file_diversity(
            rerank_results,
            file_keys_from_candidates(diversity_keys),
            self.config.RERANK_TOP_N,
            self.config.RERANK_MAX_PER_FILE,
        )

        # 提取引用信息（取重排后前 N 条；先收集索引，稍后与生成并行构建）
        final_contexts = []
        ref_indices = []
        if verbose:
            print(f"[重排序阶段] 保留前 {len(rerank_results)} 个最相关片段\n")
            for i, item in enumerate(rerank_results, 1):
                print(f"  [{i}] 相关性分数: {item['relevance_score']:.4f}")
                doc_text = item.get("document")
                idx = item["index"]
                if not doc_text:
                    doc_text = candidate_docs[idx] if idx < len(candidate_docs) else ""
                final_contexts.append(doc_text)
                ref_indices.append(idx)
        else:
            for item in rerank_results:
                doc_text = item.get("document")
                idx = item["index"]
                if not doc_text:
                    doc_text = candidate_docs[idx] if idx < len(candidate_docs) else ""
                final_contexts.append(doc_text)
                ref_indices.append(idx)

        # 生成上下文剪枝：Rerank 后仅保留前 GENERATOR_CONTEXT_TOP_N 条喂生成，压 prefill 耗时（引用仍按 RERANK_TOP_N 全量构建）
        gen_top_n = getattr(self.config, "GENERATOR_CONTEXT_TOP_N", 5) or 0
        if gen_top_n > 0 and len(final_contexts) > gen_top_n:
            if verbose:
                print(f"[生成上下文剪枝] {len(final_contexts)} -> {gen_top_n} 条（引用仍保留 {len(ref_indices)} 条）")
            final_contexts = final_contexts[:gen_top_n]

        # 4. 生成回答：后台线程流式生成，引用构建与其并行，缩短首 token 前耗时
        if verbose:
            print("\n[生成回答]\n")
        gen_box: Dict[str, Any] = {}

        def _generate_worker() -> None:
            try:
                gen_box["text"] = self.generator.generate(
                    query=question,
                    contexts=final_contexts,
                    stream=stream_callback is not None,
                    on_chunk=stream_callback,
                )
            except Exception as e:  # noqa: BLE001
                gen_box["error"] = e

        gen_thread = threading.Thread(target=_generate_worker, daemon=True)
        gen_thread.start()

        # 引用构建与生成并行（内部多线程 + 图片标题缓存，避免 LLM 调用拖慢首 token）
        with ThreadPoolExecutor(max_workers=4) as ref_executor:
            futures = [
                ref_executor.submit(self._build_reference_for_doc, idx, search_results, candidate_docs, aggregated_meta, agg_index_map)
                for idx in ref_indices
            ]
            references = [f.result() for f in futures]

        gen_thread.join()
        if "error" in gen_box:
            raise gen_box["error"]
        answer_text = gen_box["text"]

        # 5. 引用核验与过滤（拦截缺失文件引用）
        references = self._filter_citations(references, verbose=verbose)

        return {
                "content": answer_text,
                "image": [],
                "references": references
            }

