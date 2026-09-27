"""cli.py - RAG + SQL 智能问数 CLI 入口

仅负责环境初始化并委托 scripts.interactive.main，具体逻辑见 scripts/。

使用方式: python cli.py
"""

import logging
import sys

from dotenv import load_dotenv
from scripts.interactive import main as interactive_main

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


if __name__ == "__main__":
    interactive_main()
