from pathlib import Path

# 要忽略的目录（不区分大小写）
IGNORE_DIRS = {
    '.venv', '__pycache__', '.git', 'logs', 'qdrant_db', 
    '.ipynb_checkpoints', '.pytest_cache', 'node_modules', 
    'dist', 'build', 'Lib', 'Include', 'Scripts'  # 虚拟环境内部目录
}

# 只列出我们关心的代码/配置文件
EXTENSIONS = {
    '.py', '.yml', '.yaml', '.toml', '.ini', '.env', 
    '.txt', '.md', '.json', '.sql', '.xlsx', '.csv',
    '.conf', '.dockerignore', '.xml'  # 你可能用到的
}

root = Path('.')
file_list = []

for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    # 检查路径中是否包含忽略目录
    if any(part in IGNORE_DIRS for part in path.parts):
        continue
    if path.suffix in EXTENSIONS:
        file_list.append(path.as_posix())  # 用正斜杠，不乱码

# 打印结果
for f in file_list:
    print(f)