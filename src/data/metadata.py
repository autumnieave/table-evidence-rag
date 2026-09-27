import pandas as pd
import re
from pathlib import Path


def load_excel_metadata_by_title(excel_path: str):
    """
    从Excel加载研报元数据，以 title 为键，值为该标题对应的所有元数据记录列表。
    """
    if not Path(excel_path).exists():
        return {}
    
    df = pd.read_excel(excel_path)
    metadata_dict = {}
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title or title == "nan":
            continue
        clean_title = re.sub(r'[\\/*?:"<>|\n\r\t]', '', title)
        record = {
            "title": title,
            "stockName": row.get("stockName", ""),
            "stockCode": row.get("stockCode", ""),
            "orgName": row.get("orgName", ""),
            "orgSName": row.get("orgSName", ""),
            "publishDate": row.get("publishDate", ""),
            "emRatingName": row.get("emRatingName", ""),
            "lastEmRatingName": row.get("lastEmRatingName", ""),
            "indvInduName": row.get("indvInduName", ""),
            "researcher": row.get("researcher", ""),
            "market": row.get("market", ""),
        }
        metadata_dict.setdefault(clean_title, []).append(record)
    
    return metadata_dict


def get_best_metadata_for_title(title: str, metadata_dict: dict):
    """
    根据标题获取最佳元数据记录。
    若有多条，优先选择发布日期最近的；若都没有日期，选第一条。
    """
    clean_title = re.sub(r'[\\/*?:"<>|\n\r\t]', '', title)
    records = metadata_dict.get(clean_title, [])
    if not records:
        return None
    sorted_records = sorted(records, key=lambda x: str(x.get("publishDate", "")), reverse=True)
    return sorted_records[0]
