"""RAGConfig - RAG 系统统一配置类（2026-08 合并 LangChainConfig 后的唯一配置源）

所有参数均支持环境变量覆盖，也可在实例化时传入。
组件工厂方法与全局单例 get_config() 亦集中于此。
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from langchain_core.embeddings import Embeddings

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    """读取环境变量，未设置时返回默认值"""
    return os.getenv(name, default)


def _env_optional_float(name: str) -> Optional[float]:
    """读取可选浮点环境变量；未设置或为空时返回 None（不猜默认单价）。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法浮点数，按未登记处理", name, raw)
        return None


def _env_optional_bool(name: str) -> Optional[bool]:
    """读取可选布尔环境变量；未设置时返回 None（由调用方决定自动策略）。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class EmbeddingClientAdapter(Embeddings):
    """将 EmbeddingClient 包装为 LangChain Embeddings 接口，统一使用 text_type=query"""

    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.generate_embeddings(texts, text_type="query")

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.client.generate_embeddings([text], text_type="query")
        return embeddings[0]

@dataclass
class RAGConfig:
    """
    RAG 系统配置类

    路径 / 模型 / 检索 / 引用校验等参数统一在此维护；
    历史遗留的 config.langchain_config.LangChainConfig 已合并入本类（兼容别名保留至下一阶段）。
    """

    # ── 路径配置 ─────────────────────────────────────────────────────
    MARKDOWN_DIR: str = field(default_factory=lambda: _env("MARKDOWN_DIR", "测试数据/附件5：研报数据/个股研报-解析结果-完整版-2.0"))
    """个股研报 Markdown 文件夹路径"""
    EXCEL_METADATA_PATH: str = field(default_factory=lambda: _env("EXCEL_METADATA_PATH", "测试数据/附件5：研报数据/医疗服务_个股_研报信息.xlsx"))
    """个股研报元数据 Excel 路径"""
    INDUSTRY_MARKDOWN_DIR: str = field(default_factory=lambda: _env("INDUSTRY_MARKDOWN_DIR", "测试数据/附件5：研报数据/行业研报-解析结果-完整版-2.0"))
    """行业研报 Markdown 文件夹路径"""
    INDUSTRY_EXCEL_PATH: str = field(default_factory=lambda: _env("INDUSTRY_EXCEL_PATH", "测试数据/附件5：研报数据/医疗服务_行业_研报信息.xlsx"))
    """行业研报元数据 Excel 路径"""
    NEW_STOCK_DIR: str = field(default_factory=lambda: _env("NEW_STOCK_DIR", "测试数据/附件5：研报数据/个股研报-解析结果-完整版-2.0"))
    """增量插入：个股研报目录"""
    NEW_INDUSTRY_DIR: str = field(default_factory=lambda: _env("NEW_INDUSTRY_DIR", "测试数据/附件5：研报数据/行业研报-解析结果-完整版-2.0"))
    """增量插入：行业研报目录"""

    # ── Qdrant 向量库 ────────────────────────────────────────────────
    QDRANT_PERSIST_PATH: str = field(default_factory=lambda: _env("QDRANT_PERSIST_PATH", "./qdrant_db"))
    """Qdrant 本地持久化路径（本地模式）"""
    QDRANT_HOST: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    """Qdrant 服务地址"""
    QDRANT_PORT: int = field(default_factory=lambda: int(_env("QDRANT_PORT", "6333")))
    """Qdrant 服务端口"""
    QDRANT_COLLECTION_NAME: str = field(default_factory=lambda: _env("QDRANT_COLLECTION_NAME", "research_reports_v3_full"))
    """Qdrant 集合名称"""
    DISTANCE_METRIC: str = field(default_factory=lambda: _env("DISTANCE_METRIC", "COSINE"))
    """向量距离计算方式 (COSINE / DOT / EUCLIDEAN / MANHATTAN)"""

    # ── API 配置 ─────────────────────────────────────────────────────
    DASHSCOPE_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY") or None)
    """阿里云百炼 API Key"""
    API_KEYS: str = field(default_factory=lambda: _env("API_KEYS", ""))
    """允许访问 API 的 key，逗号分隔；为空时本地开发不启用鉴权"""
    API_AUTH_ENABLED: Optional[bool] = field(default_factory=lambda: _env_optional_bool("API_AUTH_ENABLED"))
    """显式鉴权开关；未设置时自动策略=API_KEYS 非空则启用"""
    API_RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: int(_env("API_RATE_LIMIT_PER_MINUTE", "10")))
    """每个 key 每分钟允许的请求数（单进程内存滑动窗口）"""
    API_AUTH_EXEMPT_PATHS: str = field(default_factory=lambda: _env("API_AUTH_EXEMPT_PATHS", "/health,/docs,/redoc,/openapi.json"))
    """免鉴权路径，逗号分隔"""
    AGENT_PLANNER_BACKEND: str = field(default_factory=lambda: _env("AGENT_PLANNER_BACKEND", "handwritten"))
    """Agent 编排后端：handwritten=自研循环（默认）/ langgraph=StateGraph 版（实验）"""
    AGENT_ENABLE_THINKING: bool = field(default_factory=lambda: _env("AGENT_ENABLE_THINKING", "false").lower() == "true")
    """Agent 编排循环是否开启思考模式（qwen3.5-plus 推理模型：默认关闭避免耗尽 max_tokens，env AGENT_ENABLE_THINKING=true 可开启）"""
    AGENT_LANGGRAPH_CHECKPOINT: bool = field(default_factory=lambda: _env("AGENT_LANGGRAPH_CHECKPOINT", "true").lower() == "true")
    """LangGraph 后端是否启用 checkpointer（thread_id=user_id 会话记忆持久化；仅 langgraph 编排后端生效，env AGENT_LANGGRAPH_CHECKPOINT=false 可关）"""
    AGENT_LANGGRAPH_CHECKPOINT_BACKEND: str = field(default_factory=lambda: _env("AGENT_LANGGRAPH_CHECKPOINT_BACKEND", "sqlite"))
    """LangGraph checkpoint 后端：sqlite=落盘（默认，需 langgraph-checkpoint-sqlite 包）/ memory=进程内存 / none=关闭"""
    AGENT_LANGGRAPH_CHECKPOINT_PATH: str = field(default_factory=lambda: _env("AGENT_LANGGRAPH_CHECKPOINT_PATH", "database/langgraph_checkpoints.sqlite"))
    """LangGraph checkpoint SQLite 文件路径（运行时数据，已 gitignore）"""
    AGENT_LANGGRAPH_MAX_HISTORY: int = field(default_factory=lambda: int(_env("AGENT_LANGGRAPH_MAX_HISTORY", "40")))
    """LangGraph checkpoint 读回的历史消息条数上限（保留首条 system 提示词）"""
    AGENT_LANGGRAPH_MULTI_AGENT: bool = field(default_factory=lambda: _env("AGENT_LANGGRAPH_MULTI_AGENT", "false").lower() == "true")
    """LangGraph 多 Agent 协作开关（supervisor-workers：规划→财务/研报子 Agent→汇总；默认关闭，实验，需配合 AGENT_PLANNER_BACKEND=langgraph）"""
    AGENT_MULTI_DIRECT_RESULT: bool = field(default_factory=lambda: _env("AGENT_MULTI_DIRECT_RESULT", "true").lower() == "true")
    AGGREGATOR_MODEL: str = field(default_factory=lambda: _env("AGGREGATOR_MODEL", ""))
    """多 Agent 汇总节点专用模型（默认空=跟随 LLM_MODEL；可设 qwen-flash 等更快模型压汇总耗时，env AGGREGATOR_MODEL）"""
    SUPERVISOR_MODEL: str = field(default_factory=lambda: _env("SUPERVISOR_MODEL", ""))
    """多 Agent 规划节点（supervisor）专用模型（默认空=跟随 LLM_MODEL；可设 qwen-flash 压拆任务耗时，非法 JSON 自动回退主模型，env SUPERVISOR_MODEL）"""
    """LangGraph 多 Agent 单任务直出开关（supervisor 只拆出 1 个财务/研报任务时跳过汇总 LLM，直接透传子结果，省 1 次 LLM 调用；env AGENT_MULTI_DIRECT_RESULT=false 可关）"""

    # ── 模型配置 ─────────────────────────────────────────────────────
    LLM_MODEL: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5-plus"))
    """大语言模型名称 (DashScope)"""
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-v2"))
    """向量化模型名称 (DashScope)"""
    EMBEDDING_MAX_CHARS: int = field(default_factory=lambda: int(_env("EMBEDDING_MAX_CHARS", "2000")))
    """单条文本**嵌入前**的字符兜底上限（0 = 不限）：超过则截断后再送 API（payload 里的正文不受影响）。
    这是**兜底**（正常由 `TABLE_SLICE_MAX_CHARS` 保证不触发），防单个超长块让整批 400"""

    EMBEDDING_BATCH_SIZE: int = field(default_factory=lambda: int(_env("EMBEDDING_BATCH_SIZE", "10")))
    """Embedding 批处理大小"""
    VECTOR_DIMENSION: int = field(default_factory=lambda: int(_env("VECTOR_DIMENSION", "1536")))
    """向量维度（需与模型输出一致）"""
    RERANK_MODEL: str = field(default_factory=lambda: _env("RERANK_MODEL", "qwen3-rerank"))
    """重排序模型名称 (DashScope)"""

    # ── 可观测性 / 成本 ─────────────────────────────────────────────
    OBSERVABILITY_LOG_ENABLED: bool = field(default_factory=lambda: _env("OBSERVABILITY_LOG_ENABLED", "true").lower() == "true")
    """是否把每请求结构化日志写入 JSONL（默认开；测试/离线可关）"""
    OBSERVABILITY_LOG_DIR: str = field(default_factory=lambda: _env("OBSERVABILITY_LOG_DIR", "训练结果数据/logs"))
    """结构化日志目录；按 YYYYMMDD.jsonl 分文件"""
    LLM_STREAM_INCLUDE_USAGE: bool = field(default_factory=lambda: _env("LLM_STREAM_INCLUDE_USAGE", "true").lower() == "true")
    """流式调用是否请求 usage（DashScope OpenAI 兼容接口支持时；不支持会自动回退）"""
    LLM_INPUT_PRICE_PER_1K: Optional[float] = field(default_factory=lambda: _env_optional_float("LLM_INPUT_PRICE_PER_1K"))
    """输入 token 单价（元/千 token）；未登记时日志写 unit_price_not_registered"""
    LLM_OUTPUT_PRICE_PER_1K: Optional[float] = field(default_factory=lambda: _env_optional_float("LLM_OUTPUT_PRICE_PER_1K"))
    """输出 token 单价（元/千 token）；未登记时日志写 unit_price_not_registered"""

    # ── 混合检索（#8 实验）───────────────────────────────────────────
    HYBRID_ENABLED: bool = field(default_factory=lambda: _env("HYBRID_ENABLED", "true").lower() == "true")
    """是否默认启用混合检索（向量 + BM25），interactive 可用 hybrid on/off 切换（默认开启：实测 K=50+混合 为精排后最优组合）"""
    HYBRID_RRF_K: int = field(default_factory=lambda: int(_env("HYBRID_RRF_K", "60")))
    """RRF 融合常数：score = Σ 1/(k + rank)"""
    HYBRID_TOPK_VECTOR: int = field(default_factory=lambda: int(_env("HYBRID_TOPK_VECTOR", "200")))
    """混合检索：向量路召回量"""
    HYBRID_TOPK_BM25: int = field(default_factory=lambda: int(_env("HYBRID_TOPK_BM25", "200")))
    """混合检索：BM25 路召回量"""
    HYBRID_VECTOR_FLOOR_RATIO: float = field(default_factory=lambda: float(_env("HYBRID_VECTOR_FLOOR_RATIO", "0.95")))
    """混合检索：融合结果中向量路文档保底比例（0=纯 RRF；0.8=top-K 至少 80% 来自向量路）"""
    BM25_INDEX_PATH: str = field(default_factory=lambda: _env("BM25_INDEX_PATH", "database/bm25_index.pkl"))
    """BM25 索引缓存路径（实际文件按 集合名+点数 后缀区分，自动重建）"""
    TABLE_BM25_INDEX_PATH: str = field(default_factory=lambda: _env("TABLE_BM25_INDEX_PATH", "database/table_bm25_index.pkl"))
    """表级 BM25 索引缓存路径（多阶段检索用，按 集合名+点数 后缀区分）"""
    MULTI_STAGE_RETRIEVAL_ENABLED: bool = field(default_factory=lambda: _env("MULTI_STAGE_RETRIEVAL_ENABLED", "false").lower() == "true")
    """是否启用多阶段检索运行时（默认关；开启后 API 启动预热表级 BM25/词表/检索链路）"""
    SIBLING_TABLE_RECALL_ENABLED: bool = field(default_factory=lambda: _env("SIBLING_TABLE_RECALL_ENABLED", "false").lower() == "true")
    """同文档兄弟表召回（默认关；同文件已选表不覆盖查询指标/期间/单位时，用同文件兄弟表替换最低排名表）"""
    MULTI_STAGE_WARMUP_ENABLED: bool = field(default_factory=lambda: _env("MULTI_STAGE_WARMUP_ENABLED", "true").lower() == "true")
    """多阶段检索启用时是否在启动阶段预热（关闭则首个请求承担冷启动）"""
    MULTI_STAGE_WARMUP_ROUNDS: int = field(default_factory=lambda: int(_env("MULTI_STAGE_WARMUP_ROUNDS", "3")))
    """预热轮数（每轮跑 通用/实体/短查询 三种形态；复用查询向量，不额外调 embedding）"""

    # ── 分块配置 ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1024")))
    """文本分块大小 (字符数)"""
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    """分块重叠大小 (字符数)"""
    CHUNK_MIN_CHARS: int = field(default_factory=lambda: int(_env("CHUNK_MIN_CHARS", "200")))
    """分块最小长度 (字符数)：过短的相邻文本块会被合并，避免单字碎片入库；0 = 关闭（旧行为）。改这里必须同步 src/data/splitter.py 的 CHUNK_MIN_CHARS 并重建索引"""
    TABLE_CHUNK_MODE: str = field(default_factory=lambda: _env("TABLE_CHUNK_MODE", "legacy"))
    """表格入库口径：legacy=一行一个 chunk（现状，线上 v3_full 口径）；structured=一张表一个/若干 chunk（重构，需重建索引）"""
    TABLE_ROW_FACT_MODE: str = field(default_factory=lambda: _env("TABLE_ROW_FACT_MODE", "off"))
    """行级事实口径：off=只写 anchor_cells（默认，不影响现有检索）；chunks=额外生成 row_fact 子块（阶段 2/3，需重建索引）"""
    TABLE_SLICE_ROWS: int = field(default_factory=lambda: int(_env("TABLE_SLICE_ROWS", "40")))
    """结构化口径下单表切片行数上限（超长表按「表头重复 + 每 K 行」切片）"""
    TABLE_SLICE_MAX_CHARS: int = field(default_factory=lambda: int(_env("TABLE_SLICE_MAX_CHARS", "1200")))
    """结构化口径下单表切片**字符**上限（0 = 不限）。嵌入 API（DashScope text-embedding-v2）
    单条输入 ≤ 2048 token，整表 Markdown 会超限（实测 702 块 > 2048 字、最长 13,239 字）⇒ 需切片；
    **改这里需重建索引**"""
    BOILERPLATE_STRIP_ENABLED: bool = field(
        default_factory=lambda: _env("BOILERPLATE_STRIP_ENABLED", "false").lower() == "true"
    )
    """锅炉板闸门（§3.2）：true = 入库时剥离文档尾部声明/签名区（`免责声明`/`分析师声明`/`评级标准`…），
    残余锅炉板块标 `chunk_type = boilerplate` **且检索时不进候选**。
    实测（473 篇正式语料）可剥离 471 篇 / 845,202 字（占全文 6.28%）；判定见 `src/data/boilerplate.py`。
    **改这里需重建索引**；默认 false（不动线上 `v3_full` 口径）"""

    # ── 检索配置 ─────────────────────────────────────────────────────
    RETRIEVAL_K: int = field(default_factory=lambda: int(_env("RETRIEVAL_K", "50")))
    """向量检索召回的候选条数（Rerank 输入深度，默认 50；加深 K 提升召回层覆盖但 Rerank 后 top10 命中反降，见 docs/检索对比报告_rerank_*.md）"""
    RERANK_TOP_N: int = field(default_factory=lambda: int(_env("RERANK_TOP_N", "10")))
    """重排序后返回的最终条数"""
    RERANK_OVERSAMPLE: int = field(default_factory=lambda: int(_env("RERANK_OVERSAMPLE", "2")))
    """Rerank 请求条数倍率：实际请求 RERANK_TOP_N×OVERSAMPLE，再经每文件上限收敛回 RERANK_TOP_N"""
    RERANK_MAX_PER_FILE: int = field(default_factory=lambda: int(_env("RERANK_MAX_PER_FILE", "2")))
    """精排后每文件最多保留片段数（0=不限制；缓解精排把上下文压缩到少数文件导致的引用覆盖下降）"""
    ENTITY_GATE_ENABLED: bool = field(default_factory=lambda: _env("ENTITY_GATE_ENABLED", "false").lower() == "true")
    """实体硬闸门（L5-B）：用查询解析出的公司实体对候选做硬过滤。**默认关**（软约束原样保留，便于 A/B）
    背景：软约束只是置换同一集合（精排吃全部 RETRIEVAL_K 条且对输入顺序不敏感），无法移除候选 ⇒ 只有硬过滤能改变最终 top-10"""
    ENTITY_GATE_MIN_KEEP: int = field(default_factory=lambda: int(_env("ENTITY_GATE_MIN_KEEP", "0")))
    """实体硬闸门安全阈值（可选，默认 0 = 不启用）：**实体命中数**低于该值时不过滤。
    实测（2026-10-16，27 题）该闸门不划算——min_keep=3 有 11 题跳过过滤、他公司占比 0.1333，
    而 min_keep=0 他公司占比 0.0 且本主体命中率相同；它也不能保护多主体对比题。
    上下文安全由 ENTITY_GATE_KEEP_UNKNOWN=true 兜底（实测空上下文 0/27）"""
    ENTITY_GATE_KEEP_UNKNOWN: bool = field(default_factory=lambda: _env("ENTITY_GATE_KEEP_UNKNOWN", "true").lower() == "true")
    """是否保留**无实体标注**的候选（行业研报等）。**默认 true 且必须为 true**：27 条点名公司题里 7 条的公司不在语料内，
    关掉会把 7/27 题的上下文清空（2026-10-16 实测；闭词表修好后 B2009 不再被清空，故由 8 降为 7）"""
    ENTITY_GATE_VOCAB_ENABLED: bool = field(default_factory=lambda: _env("ENTITY_GATE_VOCAB_ENABLED", "true").lower() == "true")
    """是否启用闭词表简称归一（`三金` → `桂林三金`）；词表优先取**向量库内实际语料**的去重 stockName（与检索集合自洽），
    取不到时降级到 config.EXCEL_METADATA_PATH（个股研报元数据）"""

    ROW_GATE_ENABLED: bool = field(default_factory=lambda: _env("ROW_GATE_ENABLED", "false").lower() == "true")
    """裸表行闸门（B-59 A）：排除「提到公司但不含指标词」的表格碎片（如 `300181. SZ 佐力药业 0.72 2.73 …`）。
    **默认关**。背景：全库 86% 的 chunk 不足 200 字，含「公司名+表行数字」的 864 条（1.5%）劫持了 4 道
    「公司+财务指标」题 top-10 的 36/40（2026-10-16 实测，`docs/评估报告/B-59_精排丢弃4条_人工复核工作单_20261016.md`）"""
    ROW_GATE_MAX_CHARS: int = field(default_factory=lambda: int(_env("ROW_GATE_MAX_CHARS", "200")))
    """裸表行闸门的长度上限：超过该长度视为正文，一律保留（默认 200，与 `filters.row_gate.MAX_ROW_CHARS` 一致）"""

    TABLE_AGG_TOPK: int = field(default_factory=lambda: int(_env("TABLE_AGG_TOPK", "20")))
    """表聚合作用范围：仅对排序后前 TOPK 条候选中的表格行拉取父表（0=不限制；默认 20 与 Rerank oversample 对齐，避免低相关表格行拖慢首查）"""

    # ── 生成配置 ─────────────────────────────────────────────────────
    TEMPERATURE: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.7")))
    """生成温度（原 LangChainConfig.LLM_TEMPERATURE 已并入）"""
    MAX_TOKENS: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "2048")))
    """最大生成 Token 数（原 LangChainConfig.LLM_MAX_TOKENS 已并入）"""
    GENERATOR_CONTEXT_TOP_N: int = field(default_factory=lambda: int(_env("GENERATOR_CONTEXT_TOP_N", "10")))
    """生成阶段使用的前置上下文条数（Rerank 后仍保留 RERANK_TOP_N 条用于引用，生成只喂前 N 条压 prefill 耗时；env GENERATOR_CONTEXT_TOP_N）"""
    STREAM: bool = field(default_factory=lambda: _env("STREAM", "true").lower() == "true")
    """是否流式输出"""

    # ── API 调用 ─────────────────────────────────────────────────────
    MAX_RETRIES: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "3")))
    """API 调用最大重试次数"""
    REQUEST_TIMEOUT: int = field(default_factory=lambda: int(_env("REQUEST_TIMEOUT", "120")))
    """外部 API 请求超时 (秒)"""
    LLM_TIMEOUT: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT", "300")))
    """LLM 生成请求超时 (秒)：qwen3.5-plus 长上下文生成可能超过默认 60s，回归期实测需 300s"""

    # ── 查询缓存与并行（路线 1：并行 + 缓存）─────────────────────────
    QUERY_CACHE_ENABLED: bool = field(default_factory=lambda: _env("QUERY_CACHE_ENABLED", "true").lower() == "true")
    """查询结果缓存开关（财务查询 / 前端 API 结果；做"修复后真实重跑"回归时设 env QUERY_CACHE_ENABLED=false 或清空缓存库，避免命中旧结果）"""
    QUERY_CACHE_DB: str = field(default_factory=lambda: _env("QUERY_CACHE_DB", "database/query_cache.db"))
    """查询缓存 SQLite 路径（运行时数据，建议 gitignore）"""
    QUERY_CACHE_TTL: int = field(default_factory=lambda: int(_env("QUERY_CACHE_TTL", "86400")))
    QUERY_CACHE_VERSION: str = field(default_factory=lambda: _env("QUERY_CACHE_VERSION", ""))
    """缓存版本号（默认空）：修改 SQL 生成/分析提示词或 FINANCIAL_PROMPT_VERSION 后 bump 该值（env QUERY_CACHE_VERSION=v2），强制旧缓存失效，避免命中过期 SQL"""
    """查询缓存 TTL（秒），默认 24h；改提示词规则后需清缓存或调小"""
    AGENT_PARALLEL_TOOLS: bool = field(default_factory=lambda: _env("AGENT_PARALLEL_TOOLS", "true").lower() == "true")
    """Agent 工具/子任务是否并行执行（自研 AgentPlanner 同轮多工具、LangGraph 多 Agent 子任务）"""
    # ── SQL 校验（Agent 工具循环三层防线）────────────────────────────
    AGENT_SQL_VALIDATE: bool = field(default_factory=lambda: _env("AGENT_SQL_VALIDATE", "true").lower() == "true")
    """SQL 生成是否做静态+编译校验（false=关闭，直接直通）"""
    AGENT_NATIVE_RETRY: int = field(default_factory=lambda: int(_env("AGENT_NATIVE_RETRY", "2")))
    """原生 SQL 生成校验失败后的重试次数（复用三层防线：静态校验 + MySQL 编译）"""
    AGENT_METRIC_STANDARDIZE: bool = field(default_factory=lambda: _env("AGENT_METRIC_STANDARDIZE", "true").lower() == "true")
    """SQL 生成前是否先做指标标准化小调用（问题→JSON，B-12 两步分解第 1 步；env AGENT_METRIC_STANDARDIZE=false 关闭并回退旧自选路径）"""
    AGENT_DYNAMIC_FEWSHOT: bool = field(default_factory=lambda: _env("AGENT_DYNAMIC_FEWSHOT", "false").lower() == "true")
    """SQL 生成是否启用动态 few-shot 示例库（B-16：命中 prompts/examples 注入 SQL_GEN；默认关以便与静态基线对照与回退，env AGENT_DYNAMIC_FEWSHOT=true 开启）"""
    AGENT_FEWSHOT_MODE: str = field(default_factory=lambda: _env("AGENT_FEWSHOT_MODE", "").strip().lower())
    """SQL 生成 few-shot 三态开关（B-39 三组对照）：none=不注入示例 | static=固定示例（不检索）| dynamic=按题型检索注入；留空则回退旧布尔开关 AGENT_DYNAMIC_FEWSHOT（true→dynamic / false→none）"""
    AGENT_FEWSHOT_STATIC_K: int = field(default_factory=lambda: int(_env("AGENT_FEWSHOT_STATIC_K", "2")))
    """static 模式固定示例条数（默认 2，与 dynamic 的 k=2 对齐以控制 token 量可比）"""
    AGENT_FEWSHOT_STATIC_STRATEGY: str = field(default_factory=lambda: _env("AGENT_FEWSHOT_STATIC_STRATEGY", "per_type").strip().lower())
    """static 模式固定示例取法：per_type=每题型各 1 条（默认，类型多样但不检索）| head=按题型文件序取前 k 条"""
    MYSQL_HOST: str = field(default_factory=lambda: _env("MYSQL_HOST", "127.0.0.1"))
    """MySQL 地址（SQL 校验 schema / 编译终审用）"""
    MYSQL_PORT: int = field(default_factory=lambda: int(_env("MYSQL_PORT", "3306")))
    """MySQL 端口"""
    MYSQL_USER: str = field(default_factory=lambda: _env("MYSQL_USER", "root"))
    """MySQL 用户名"""
    MYSQL_PASSWORD: str = field(default_factory=lambda: _env("MYSQL_PASSWORD", "123456"))
    """MySQL 密码（生产环境请用环境变量覆盖）"""
    MYSQL_DATABASE: str = field(default_factory=lambda: _env("MYSQL_DATABASE", "financial_database"))
    """MySQL 库名（含 4 张财务白名单表）"""

    # ── 多轮对话 ─────────────────────────────────────────────────────
    ENABLE_MULTI_TURN: bool = field(default_factory=lambda: _env("ENABLE_MULTI_TURN", "false").lower() == "true")
    """是否启用多轮对话上下文管理"""
    CONVERSATION_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(_env("CONVERSATION_TIMEOUT_SECONDS", "300")))
    """会话超时时间 (秒)"""
    MEMORY_BACKEND: str = field(default_factory=lambda: _env("MEMORY_BACKEND", "sqlite"))
    """会话记忆后端: sqlite（默认）/ redis / none（none=纯内存）"""
    MEMORY_DB_PATH: str = field(default_factory=lambda: _env("MEMORY_DB_PATH", "database/chat_memory.db"))
    """SQLite 记忆库路径"""
    MEMORY_REDIS_URL: str = field(default_factory=lambda: _env("MEMORY_REDIS_URL", "redis://localhost:6379/0"))
    """Redis 记忆连接串"""
    MEMORY_TTL_SECONDS: int = field(default_factory=lambda: int(_env("MEMORY_TTL_SECONDS", "86400")))
    """会话快照保留时长 (秒)，默认 24h"""

    # ── 引用校验（L1）配置 ────────────────────────────────────────────
    CITATION_CORPUS_ROOT: str = field(default_factory=lambda: _env("CITATION_CORPUS_ROOT", "data_corpus/全部数据/正式数据/附件5：研报数据"))
    """引用核验语料根目录"""
    CITATION_MATCH_MODE: str = field(default_factory=lambda: _env("CITATION_MATCH_MODE", "comma"))
    """数字匹配口径：raw=原样 / comma=逗号归一化 / loose=comma+去空白"""
    CITATION_CHECK_ON_QUERY: bool = field(default_factory=lambda: _env("CITATION_CHECK_ON_QUERY", "true").lower() == "true")
    """在线查询是否对引用执行 L1 核验"""
    CITATION_FILTER_MISSING: bool = field(default_factory=lambda: _env("CITATION_FILTER_MISSING", "true").lower() == "true")
    """核验后是否过滤文件缺失的引用（拦截模型生成式引用）"""
    CITATION_FILTER_ZERO_HIT: bool = field(default_factory=lambda: _env("CITATION_FILTER_ZERO_HIT", "false").lower() == "true")
    """是否过滤数字零命中引用（默认关闭，避免误杀改写摘要）"""

    # ── 校验 ─────────────────────────────────────────────────────────
    def validate(self) -> None:
        """检查必要配置，缺失时打印警告"""
        if not self.DASHSCOPE_API_KEY:
            logger.warning("DASHSCOPE_API_KEY 未设置，请通过环境变量或 .env 文件配置")

    # ── 组件工厂（自 config.langchain_config 合并）────────────────────
    def get_chat_model(self):
        """获取 DashScope Chat Model 实例"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.LLM_MODEL,
            api_key=self.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            # 关闭思考模式，避免 reasoning 消耗全部 max_tokens 导致回答为空
            extra_body={"enable_thinking": False},
        )

    def get_embeddings(self):
        """获取 OpenAI 兼容模式的 Embeddings 实例"""
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=self.EMBEDDING_MODEL,
            api_key=self.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def get_embedding_adapter(self):
        """获取基于 EmbeddingClient 的适配器实例（延迟导入避免循环依赖）"""
        from embeddings import EmbeddingClient
        rag_config = RAGConfig(
            DASHSCOPE_API_KEY=self.DASHSCOPE_API_KEY,
            EMBEDDING_MODEL=self.EMBEDDING_MODEL,
        )
        client = EmbeddingClient(rag_config)
        return EmbeddingClientAdapter(client)

    def get_vector_store(self, collection_name: Optional[str] = None, embedding=None):
        """获取 Qdrant VectorStore 实例（from_existing_collection 路径）"""
        from langchain_qdrant import QdrantVectorStore
        embeddings = embedding if embedding is not None else self.get_embeddings()
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name or self.QDRANT_COLLECTION_NAME,
            host=self.QDRANT_HOST,
            port=self.QDRANT_PORT,
            content_payload_key="content",
        )

    def get_vector_store_direct(self, client, collection_name: str, embedding):
        """通过 QdrantVectorStore 构造函数创建实例，跳过 from_existing_collection 的维度验证"""
        from langchain_qdrant import QdrantVectorStore
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            content_payload_key="content",
            validate_collection_config=False,
        )

    def get_text_splitter(self):
        """获取 LangChain 文本分块器"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    def get_rerank(self):
        """获取 DashScope Rerank 实例"""
        from langchain_community.document_compressors import DashScopeRerank
        return DashScopeRerank(
            model=self.RERANK_MODEL,
            dashscope_api_key=self.DASHSCOPE_API_KEY,
            top_n=self.RERANK_TOP_N,
        )


# ======================================================================
# 模块级单例
# ======================================================================
_config_instance: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """获取全局单例配置（避免重复创建）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig()
        _config_instance.validate()
    return _config_instance
