FROM python:3.11-slim

WORKDIR /app

# 系统依赖（Qdrant 客户端扩展编译等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码（#4 入口收敛：uvicorn app.api:app；⑤ 收口后业务包在 src/，根目录留 CLI 启动器与 eval/scripts）
# 大数据目录经 .dockerignore 排除，需通过卷挂载或宿主机构建索引
COPY . .

# 输出目录
RUN mkdir -p result database

EXPOSE 8000

# 启动命令（FastAPI 后端）
CMD ["uvicorn", "app.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
