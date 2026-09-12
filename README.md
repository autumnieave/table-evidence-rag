# 上市公司“智能问数”助手系统

面向上市公司研报与财报的端到端智能问答系统：用户用自然语言即可查询财务数据、研报观点，答案带引用可溯源。采用 **LangGraph 多 Agent 编排 + SQL 财务链路 + RAG 研报链路**，配套 FastAPI / React 前端与完整评估闭环。

[![CI](https://github.com/autumnieave/smart-financial-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/autumnieave/smart-financial-qa/actions/workflows/ci.yml) [![tests](https://img.shields.io/badge/473%20tests-passing-brightgreen)]() [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 核心指标

- **SQL 编译通过率：96.9% → 100%**（80 题全量回归；最新口径 **103/103** @2026-09-10 提示词 v10，有 SQL 题 69/69 全通过、失败 0；历史口径 Agent 224/224、原生 SQL 102/102、137/137@2026-09-05，分母各不相同，不可直接对比）
- **引用文件可溯源 100%（1080/1080）**、**答案数字可溯源 99.9%（4068/4071，归一化口径；3 项未溯源为非数据 token）**，人工回查真实幻觉 **0 例**
- **473 个离线单测全部通过**（473 用例 / 37 个测试文件，零外部依赖，CI 自动执行）
- 数字级引用命中率 **70.2% → 74.9%**（混合检索：向量 + BM25 + RRF）

## 功能亮点

- **LangGraph supervisor-workers 多 Agent 编排**：supervisor 拆解任务，财务（SQL）/ 研报（RAG）子 Agent 并行取数，单任务直出、多任务聚合；条件边路由 + checkpoint 按 `user_id` 持久化会话；自研手写 RAG 与多轮澄清链路保留，供 CLI 本地回归使用
- **SQL 生成质量闭环**：自然语言 → Schema + 字段白名单 → 静态校验 → MySQL 试运行（15s 超时）→ 执行 + 自动分析与 ECharts 图表，失败自动带错误重试
- **混合检索**：Qdrant 向量 + BM25 关键词 + RRF 融合，经 `qwen3-rerank` 精排后生成
- **L1 引用核验器**：自动核验答案数字与引用文件的对应关系，端到端可溯源
- **记忆持久化**：SQLite 默认 / Redis 可选，按 `user_id` 存取，服务重启后上下文可恢复
- **全栈可部署**：FastAPI（REST + SSE 流式）+ React 19 + Qdrant + Docker Compose

## 架构图

```mermaid
flowchart TD
    subgraph DATA["数据与索引层"]
        A["研报 Markdown + 财报 PDF"] --> B["MinerU 解析 + 层级分块\noverlap=100"]
        B --> C["Embedding\ntext-embedding-v2"]
        C --> D[("Qdrant\nresearch_reports_v3_full")]
        E["财报字段抽取"] --> F[("MySQL\nfinancial_database")]
    end

    subgraph ONLINE["在线问答层"]
        Q["用户问题"] --> S["LangGraph supervisor\n任务拆解 + 条件边路由"]
        S -->|"财务问题"| T1["财务子 Agent\nSQL 三层防线 → MySQL → ECharts"]
        S -->|"研报问题"| T2["研报子 Agent\n混合检索 → Rerank → 生成"]
        T1 --> AGG["聚合节点"]
        T2 --> AGG
        AGG --> R["答案 + 引用（L1 核验）"]
    end

    subgraph MEM["会话层"]
        M["checkpoint 按 user_id 持久化\n多轮澄清"] -.-> Q
    end
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可复现构建（版本锁定）：pip install -r requirements.lock.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
DASHSCOPE_API_KEY=sk-xxx        # 阿里云百炼 DashScope（必填）
MYSQL_HOST=127.0.0.1            # MySQL 财务库（原生财务查询链路，非 Docker 模式需自备）
```

### 3. 启动 Qdrant 并构建索引

```bash
docker compose up -d qdrant     # 或本地 Qdrant（localhost:6333）
python rag_全流程构建.py --build
```

### 4. 启动服务

```bash
# 交互式问答
python rag_全流程构建.py

# Web 后端（端口 8000）
uvicorn app.api:app --reload --port 8000

# 前端（qa-frontend 目录，端口 5173）
cd qa-frontend && npm install && npm run dev

# 或 Docker Compose 一键启动（Qdrant + 后端 + Nginx 前端，访问 http://localhost:8080）
# 注意：compose 内不含 MySQL 服务，后端连宿主已导入数据的 MySQL（host.docker.internal:3306，可在 .env 改 MYSQL_*）；
# 全新环境无外部 MySQL 时，请先按 docs/DEPLOYMENT.md 自备 MySQL，或在 docker-compose.yml 中自行补 mysql 服务块
docker compose up -d --build
```

{anchor}

### 数据与评估资产说明

- **原始数据不随仓库分发**（原始数据，按版权不公开）：研报语料按 `docs/DEPLOYMENT.md` 放置后运行 `python rag_全流程构建.py --build` 构建索引；财务数据由公开财报经 `tools/data_scripts/pdf处理+校验入库.py` 抽取入库（表结构见 `database/schema.sql`，仅建表、不含数据）。
- **评估资产为本地 gitignored 资产，不入库**：golden 快照（`database/golden/`，80 题/108 子问题评估基准）、字段抽取记录（`database/extracted_missing_fields.csv`）及各回归明细 JSON 仅保存在本地，用于复现文档中的评估口径。
- **可复现范围**：源码、离线单测（零外部依赖）、CI 与评估框架完整入库，clone 后即可运行；完整数据与评估基准需按文档自备。
- **安全提示**：MySQL 默认密码（`MYSQL_ROOT_PASSWORD` / `MYSQL_PASSWORD` = `123456`）仅用于本地开发，生产/公网部署务必通过 `.env` 修改。


## 交互命令

| 命令 | 说明 |
| :--- | :--- |
| `agent on/off` | 开关 Agent 多步推理（默认 LangGraph multi-agent） |
| `hybrid on/off` | 切换混合检索（向量 + BM25） |
| `multi-turn on/off` | 多轮澄清对话 |
| `status` | 查看各模式开关状态 |
| `rebuild` | 强制重建索引 |
| `addstock` / `addindustry` | 增量插入个股 / 行业研报 |
| `new` | 开启新话题（重置会话状态） |

## 测试与评估

```bash
python -m pytest tests/ -q        # 473 个离线单测 / 37 个测试文件（零外部依赖）
python -m eval sql --suite full   # SQL 全量回归（需本地 golden 数据）
python -m eval citation           # L1 引用核验（需本地语料）
python -m eval report             # 聚合评估报告
```

> 说明：`python -m eval` 依赖本地评估资产（golden 快照等，不入库），缺失时仅影响评估复现，不影响系统运行；单测不依赖任何外部服务与数据。

- CI：`.github/workflows/ci.yml` 在 `master` push / PR 时自动执行全部单测
- 评估口径与逐题明细见 `docs/评估报告/评估报告.md`、`docs/评估报告/SQL编译修复前后对比报告.md`、`docs/问题记录/badcase_台账.md`

## 技术栈

| 类别 | 选型 |
| :--- | :--- |
| 编程语言 | Python 3.11 |
| 大模型平台 | 阿里云百炼 DashScope（`qwen3.5-plus` / `text-embedding-v2` / `qwen3-rerank`） |
| Agent 编排 | LangGraph（supervisor-workers，主链路）、Function Calling（自研，回退路径） |
| RAG 组件 | LangChain 生态子包（分块/模型适配/Qdrant 客户端）、Qdrant、BM25（纯 Python）+ RRF |
| Web 后端 | FastAPI + Uvicorn（REST + SSE 流式） |
| 前端 | React 19 + Vite + Tailwind CSS（`qa-frontend/`） |
| 部署 | Docker Compose（Qdrant + 后端 + Nginx 前端） |
| 数据/校验 | MinerU、pandas、pymysql、sqlparse |

## 项目结构

```
app/            FastAPI 入口（api.py / schemas.py）
core/           链路接口（IRetriever / IReranker / IGenerator）
eval/           评估闭环（golden / SQL / citation / report）
pipelines/      RAGPipeline 全流程编排
agents/         LangGraph 多 Agent + 自研 AgentPlanner（对照）
prompts/        唯一 Prompt 目录
tools/          SQL 校验器 / 原生财务查询 / 数据处理脚本
tests/          473 个用例 / 37 个测试文件（离线单测）
qa-frontend/    React 19 前端
```

## 相关文档

| 文档 | 内容 |
| --- | --- |
| `docs/ARCHITECTURE.md` | 目标框架 vs 现状差距清单 |
| `docs/DEPLOYMENT.md` | 部署说明（本地 / 全栈 / 常见问题） |
| `docs/评估报告/评估报告.md` | 聚合评估报告（golden / SQL / 引用核验） |
| `docs/评估报告/LangGraph对照.md` | LangGraph 对照版 Agent 验证 |
| `docs/评估报告/MultiAgent对照.md` | supervisor-workers 多 Agent 协作 |
| `docs/评估报告/SQL编译修复前后对比报告.md` | SQL 质量修复前后对比与口径 |
| `docs/评估报告/RAG引用核验报告.md` | L1 引用核验明细 |
| `docs/评估报告/overlap对比实验.md` | 分块参数 overlap 对比（100 最优） |
| `docs/评估报告/检索对比实验.md` | 检索层对比：纯向量 vs 混合检索（引用命中） |
| `docs/问题记录/badcase_台账.md` | SQL badcase 台账与修复闭环 |
| `docs/问题记录/9题端到端人工抽检记录.md` | 端到端人工抽检记录 |

## License

MIT License，见 [LICENSE](LICENSE)。
