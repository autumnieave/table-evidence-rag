"""core 包 —— 核心抽象接口与适配实现（2026-08-22 链路收敛 #3）

- interfaces.py：IRetriever / IReranker / IGenerator（Protocol 结构化接口）
- retrievers.py：HandwrittenRetriever（默认）、LangChainRetriever（实验）
- rerankers.py / generators.py：RerankClient / LLMGenerator 的接口适配层

RAGPipeline.query() 只依赖本包接口，三条链路（手写 / 检索器 / LCEL）的差异被隔离在实现处。
"""
