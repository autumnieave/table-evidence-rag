"""eval 包 —— 评估闭环（2026-08-22 评估框架 #5）

- golden.py：golden set 版本化（database/golden/ 快照 + manifest）
- metrics.py：指标聚合与报告生成（SQL 回归 / 引用核验 / badcase）
- runner.py：统一 CLI（python -m eval ...）

用法::

    python -m eval golden init --source 训练结果数据/result_3_parallel.xlsx --version v1 --tag "80题全量"
    python -m eval golden list
    python -m eval golden verify --version v1
    python -m eval sql --suite full --limit 3
    python -m eval citation --refs 训练结果数据/references_all.json
    python -m eval report
"""

__all__ = ["golden", "metrics", "runner"]
