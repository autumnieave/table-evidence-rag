"""app 包 —— FastAPI Web 入口（2026-08-22 入口收敛 #4）

收敛自根目录 app.py：
- app/api.py：FastAPI 应用与路由（uvicorn app.api:app 启动）
- app/schemas.py：请求/响应 Pydantic 模型

启动: uvicorn app.api:app --reload --port 8000
"""
