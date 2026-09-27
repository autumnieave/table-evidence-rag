import re
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.metadata import load_excel_metadata_by_title, get_best_metadata_for_title


def load_markdown_documents(directory_path: str, excel_metadata_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从指定目录递归加载所有Markdown文件

    遍历指定目录及其子目录，读取所有.md文件内容，并提取文件元数据
    """
    excel_metadata_dict = {}
    if excel_metadata_path:
        excel_metadata_dict = load_excel_metadata_by_title(excel_metadata_path)

    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory_path}")

    documents = []
    md_files = list(directory.glob("*.md"))
    for idx, md_file in enumerate(md_files, 1):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            stat = md_file.stat()
            file_title = md_file.stem

            metadata = {
                "source": str(md_file),
                "filename": md_file.name,
                "file_path": str(md_file.relative_to(directory)),
                "modified_time": stat.st_mtime,
                "file_size": stat.st_size,
                "domain": "finance",
                "type": "research",
                "title": file_title,
            }

            if excel_metadata_dict:
                matched = get_best_metadata_for_title(file_title, excel_metadata_dict)
                if matched:
                    metadata.update(matched)
                else:
                    metadata.update({
                        "stockName": "",
                        "stockCode": "",
                        "orgName": "",
                        "publishDate": "",
                        "emRatingName": "",
                        "indvInduName": "",
                        "researcher": "",
                    })

            documents.append({"content": content, "metadata": metadata})
        except Exception:
            continue

    return documents


def load_industry_documents(industry_dir: str, excel_metadata_path: str) -> List[Dict[str, Any]]:
    logger = None
    excel_metadata_dict = load_excel_metadata_by_title(excel_metadata_path)
    directory = Path(industry_dir)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {industry_dir}")

    documents = []
    md_files = list(directory.rglob("*.md"))
    for idx, md_file in enumerate(md_files, 1):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_title = md_file.stem
            metadata = {
                "source": str(md_file),
                "filename": md_file.name,
                "file_path": str(md_file.relative_to(directory)),
                "modified_time": md_file.stat().st_mtime,
                "file_size": md_file.stat().st_size,
                "domain": "finance",
                "type": "research",
                "doc_type": "industry",
                "title": file_title,
                "stockName": "",
                "stockCode": "",
                "market": "",
            }

            if excel_metadata_dict:
                matched = get_best_metadata_for_title(file_title, excel_metadata_dict)
                if matched:
                    matched["indvInduName"] = matched.get("industryName", "")
                    metadata.update(matched)
                else:
                    metadata.update({
                        "orgName": "", "orgSName": "", "publishDate": "",
                        "emRatingName": "", "lastEmRatingName": "",
                        "indvInduName": "", "researcher": ""
                    })

            documents.append({"content": content, "metadata": metadata})
        except Exception:
            continue

    return documents
