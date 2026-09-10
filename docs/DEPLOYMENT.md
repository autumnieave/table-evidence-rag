# 部署指南

> 覆盖三种形态：**本地开发**（推荐调试）、**Docker Compose 全栈**（一键上线）、**纯后端服务**。
> 对应架构差距清单 #9（2026-08-22 完成：`.env.example`、修复后端 Dockerfile、`.dockerignore`、本指南）。

## 1. 架构与端口

| 服务 | 说明 | 端口 | 依赖 |
| --- | --- | --- | --- |
| `qdrant` | 向量数据库（集合 `research_reports_v3`） | 6333 / 6334 | 无 |
| `redis` | 会话记忆（`MEMORY_BACKEND=redis` 时启用） | 6379 | 无 |
| `backend` | FastAPI（`uvicorn app.api:app`，REST + SSE） | 8000 | qdrant；redis（可选）；MySQL（财务查询） |
| `frontend` | React 19 + Nginx 反代到后端 | 8080（容器 80） | backend |

外部依赖：

- **阿里云百炼 DashScope API Key（必填）**：Embedding / Rerank / LLM 均走 DashScope
- **MySQL（财务查询）**：Agent 财务数据查询走原生 SQL 链路（`tools/native_financial.py`），需本机 `3306` 可连；不可用时返回友好错误 JSON，RAG 检索不受影响。

## 2. 前置条件

- Python 3.11+（后端）、Docker + Compose（Qdrant / Redis / 全栈）、Node 20+（仅前端构建）
- 在项目根目录准备 `.env`（复制 `.env.example`，至少填写 `DASHSCOPE_API_KEY`）

## 3. 本地开发（推荐）

```powershell
# 1) 虚拟环境与依赖
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
pip install -r requirements-dev.txt        # 可选：跑单测

# 2) 配置
Copy-Item .env.example .env                # 填写 DASHSCOPE_API_KEY

# 3) 启动 Qdrant
docker compose up -d qdrant

# 4) 构建 / 重建向量索引（首次必做）
python rag_全流程构建.py --build           # 增量构建
python rag_全流程构建.py --rebuild         # 强制重建

# 5) 启动后端
uvicorn app.api:app --reload --port 8000

# 6) 前端（可选）
cd qa-frontend
npm install
npm run dev                                # http://localhost:5173

# 7) 健康检查
Invoke-RestMethod http://localhost:8000/health
```

交互式 CLI：`python rag_全流程构建.py`，支持 `agent` / `multi-turn` / `langchain` / `chain` / `hybrid on|off` 开关切换，`status` 查看当前模式。

## 4. Docker Compose 全栈

```powershell
# 一次性构建并启动全部服务（qdrant + redis + backend + frontend）
docker compose up -d --build

# 前端入口 http://localhost:8080 ，后端文档 http://localhost:8000/docs
docker compose logs -f backend             # 查看后端日志
docker compose ps                          # 查看各服务状态
docker compose down                        # 停止（保留数据卷）
docker compose down -v                     # 停止并清理数据卷（慎用）
```

说明：

- 后端镜像通过 `.dockerignore` 排除大数据目录（`<数据目录>/`、`qdrant_storage/` 等），仅打包源码；
- `./database` 以卷挂载：SQLite 会话记忆与 BM25 索引缓存在容器外持久化；`./result` 挂载图表输出；
- 环境变量在 `.env` 中维护，compose 以 `${VAR:-default}` 注入；
- 若宿主机已有 Qdrant 数据（`qdrant_storage/`），首次启动前可先在本地执行 `python rag_全流程构建.py --build` 预构建，再把 `qdrant_storage/` 一并交给容器（或直接在容器内 build，注意 `<数据目录>/` 未入镜像，需另挂载）。

## 5. 环境变量清单

完整注释版见 `.env.example`，常用项：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 无 | **必填**，DashScope 密钥 |
| `MYSQL_HOST` / `MYSQL_PORT` | `127.0.0.1` / `3306` | MySQL 财务库（原生 SQL 链路） |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant 地址（compose 内为 `qdrant`） |
| `QDRANT_COLLECTION_NAME` | `research_reports_v3` | 向量集合名（正式数据全量语料为 `research_reports_v3_full`，重建脚本见 README） |
| `LLM_MODEL` | `qwen3.5-plus` | 生成模型 |
| `EMBEDDING_MODEL` | `text-embedding-v2` | 向量模型 |
| `RERANK_MODEL` | `qwen3-rerank` | 精排模型 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1024` / `100` | 文本分块参数（overlap=100 经 2026-08-24 对比实验确认，见 `docs/评估报告/overlap对比实验.md`） |
| `MEMORY_BACKEND` | `sqlite` | `sqlite` / `redis` / `none` |
| `MEMORY_REDIS_URL` | `redis://localhost:6379/0` | redis 后端连接串 |
| `HYBRID_ENABLED` | `true` | 混合检索（向量 + BM25 + RRF 融合）默认开关 |
| `BM25_INDEX_PATH` | `database/bm25_index.pkl` | BM25 缓存路径（按 集合名+点数 自动重建） |
| `CITATION_CORPUS_ROOT` | `<数据根目录>/全部数据/正式数据/附件5：研报数据` | L1 引用核验语料根目录（正式数据全量 473 篇） |
| `CITATION_MATCH_MODE` | `comma` | 引用数字匹配口径（raw / comma / loose） |
| `AGENT_PLANNER_BACKEND` | 代码默认 `handwritten`；**当前运行环境通过 `.env` 设为 `langgraph`** | Agent 规划器后端：`handwritten`（自研，回退路径）/ `langgraph`（LangGraph 编排）。当前环境配合 `AGENT_LANGGRAPH_MULTI_AGENT=true` 运行 supervisor-workers 多 Agent 主链路 |
| `AGENT_ENABLE_THINKING` | `false` | Agent 循环思考模式（推理模型默认关闭，避免耗尽 max_tokens） |
| `AGENT_SQL_VALIDATE` | `true` | SQL 输出守卫开关（静态校验 + MySQL 编译，失败带错误提示重问） |
| `AGENT_NATIVE_RETRY` | `2` | 原生 SQL 生成失败重试次数 |
| `LLM_TIMEOUT` | `300` | LLM 调用超时（秒，Agent 长轨迹兜底） |
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | `127.0.0.1:3306` / `root` / `financial_database` | SQL 守卫编译终审连接（schema 见 `database/schema.sql`，仅表结构，数据需自行抽取入库） |

## 6. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 启动报 Qdrant 连接失败 | 先 `docker compose up -d qdrant`；或改 `QDRANT_HOST/PORT` 指向已有服务 |
| `--build/--rebuild` 找不到研报目录 | `MARKDOWN_DIR` 等路径默认指向 `<数据目录>/`，确认该目录在宿主机存在且 `.env` 未覆盖错误 |
| Agent 财务查询报"MySQL schema/连接加载失败" | MySQL 未启动或 `MYSQL_*` 配置错误；普通 RAG 不受影响 |
| 回答为空 / 极短 | `LLM_MODEL` 为推理模型时必须关闭思考（代码已带 `enable_thinking: False`）；或检查 `MAX_TOKENS` |
| `MEMORY_BACKEND=redis` 连不上 | 确认 Redis 已启动（compose 含 redis 服务），或改回 `sqlite` |
| 前端访问 API 404 | 前端经 Nginx 反代 `/api` 到 backend；本地联调时确认 `VITE_API_URL` 与后端端口一致 |
| Docker 构建包含大文件 | `.dockerignore` 已排除 `<数据目录>/ qdrant_storage/ .venv/` 等；如仍异常检查 `.dockerignore` 是否被改动 |
| Windows 控制台中文乱码 | 使用 `python -X utf8` 或确保终端为 UTF-8；CLI 已对 stdout 做 UTF-8 reconfigure |

## 7. 数据与持久化

- **向量索引**：Qdrant 数据在 `qdrant_storage/`（compose 卷挂载）
- **会话记忆**：SQLite `database/chat_memory.db`（默认）或 Redis
- **BM25 索引缓存**：`database/bm25_index.<集合名>.<点数>.pkl`，点数变化自动重建
- **图表输出**：`result/`
- 以上运行时数据均已被 `.gitignore` 排除，不入库
