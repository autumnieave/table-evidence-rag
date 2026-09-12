# -*- coding: utf-8 -*-
"""
整合脚本：PDF 财务报表提取 + 单条数据校验 + 入库 + 重试机制
完全保留原始 pdf2csv 和 check 的核心逻辑，仅做流程封装。
"""

import pdfplumber
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import re
import os
import sys
import copy
import pymysql
from datetime import datetime
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# 全局文件写入锁
csv_lock = threading.Lock()

# ================== 配置参数（与原代码一致） ==================
OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AI_MODEL_TITLE = "qwen-long-latest"
AI_MODEL_EXTRACT = "qwen3.5-flash"

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'financial_database',
    'charset': 'utf8mb4'
}

# 数据链路目录（正式数据：附件2 财务报告）
PDF_DIRS = [
    ("sz", "B题数据及提交说明/全部数据/正式数据/附件2：财务报告/reports-深交所"),
    ("sh", "B题数据及提交说明/全部数据/正式数据/附件2：财务报告/reports-上交所"),
]
# 字段定义表（正式数据：附件3）
FIELDS_XLSX = "B题数据及提交说明/全部数据/正式数据/附件3：数据库-表名及字段说明.xlsx"

# 股票代码简称对照表（原样保留）
CODE2NAME = {
    '600080': '金花股份', '600085': '同仁堂', '600129': '太极集团',
    '600222': '太龙药业', '600252': '中恒集团', '600285': '羚锐制药',
    '600329': '达仁堂', '600332': '白云山', '600351': '亚宝药业',
    '600422': '昆药集团', '600436': '片仔癀', '600479': '千金药业',
    '600518': '康美药业', '600535': '天士力', '600557': '康缘药业',
    '600566': '济川药业', '600572': '康恩贝', '600594': '益佰制药',
    '600613': '神奇制药', '600671': 'ST目药', '600750': '江中药业',
    '600771': '广誉远', '600976': '健民集团', '600993': '马应龙',
    '603139': '康惠制药', '603439': '贵州三力', '603567': '珍宝岛',
    '603858': '步长制药', '603896': '寿仙谷', '603998': '方盛制药',
    '603127': '昭衍新药', '603259': '药明康德', '688222': '成都先导',
    '688276': '百克生物'
}
shen = {
    '002082': '万邦德','000423': '东阿阿胶', '000989': '九芝堂',
    '000538': '云南白药', '000650': '仁和药业', '002603': '以岭药业',
    '002317': '众生药业', '300181': '佐力药业', '002390': '信邦制药',
    '002907': '华森制药', '000999': '华润三九', '000790': '华神科技',
    '000590': '启迪药业', '002198': '嘉应制药', '002287': '奇正藏药',
    '002773': '康弘药业', '300869': '康泰医学', '301331': '恩威医药',
    '300158': '振东制药', '300519': '新光药业', '002873': '新天药业',
    '002219': '新里程', '002275': '桂林三金', '002107': '沃华医药',
    '002728': '特一药业', '002589': '瑞康医药', '002566': '益盛药业',
    '002864': '盘龙药业', '301111': '粤万年青', '002349': '精华制药',
    '300026': '红日药业', '300878': '维康药业', '002166': '莱茵生物',
    '002737': '葵花药业', '002424': '贵州百灵', '002898': '赛隆药业',
    '000766': '通化金马', '300391': '长药控股', '300534': '陇神戎发',
    '300147': '香雪制药', '301096': '百诚医药', '301080': '百普赛斯',
    '300244': '迪安诊断', '002821': '凯莱英', '301033': '迈普医学',
    '300347': '泰格医药'
}
NAME2CODE = {v: k for k, v in shen.items()}

def clean_table_data(table):
    cleaned_data = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append('')
            else:
                cell = str(cell).strip().replace('\n', '').replace(',', '')
                cleaned_row.append(cell)
        cleaned_data.append(cleaned_row)
    return cleaned_data

def extract_tables_from_pdf(pdf_path):
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        cleaned_table = clean_table_data(table)
                        all_tables.extend(cleaned_table)
                except Exception as e:
                    # 捕获单页解析错误，打印日志但不中断整个文件处理
                    print(f"⚠️ 解析第 {page.page_number} 页表格时发生非致命错误 (可能是P205错误)，已跳过该页: {e}")
    except Exception as e:
        print(f"❌ 打开或解析 PDF 文件失败: {e}")
        return []
    return all_tables

def extract_context(pages, current_page, match_bbox, start_keyword):
    y_top = match_bbox['top']
    y_bottom = match_bbox['bottom']
    page_height = pages[current_page].height
    page_width = pages[current_page].width

    context_top_start = max(0, y_top - 400)
    context_top_box = (0, context_top_start, page_width, y_top)
    context_top_text = pages[current_page].crop(context_top_box).extract_text()
    if len(context_top_text) < 400:
        gap = 400 - len(context_top_text)
        page_height_new = pages[current_page-1].height
        page_width_new = pages[current_page-1].width
        context_top_start = max(0, page_height_new - gap)
        context_top_box = (0, context_top_start, page_width_new, page_height_new)
        context_top_text_add = pages[current_page - 1].crop(context_top_box).extract_text() if current_page > 0 else ""
        context_top_text = context_top_text_add + context_top_text

    context_bottom_end = min(page_height, y_bottom + 1000)
    context_bottom_box = (0, y_bottom, page_width, context_bottom_end)
    context_bottom_text = pages[current_page].crop(context_bottom_box).extract_text()
    if len(context_bottom_text) < 1000:
        gap = 1000 - len(context_bottom_text)
        page_height_new = pages[current_page+1].height
        page_width_new = pages[current_page+1].width
        context_bottom_end = min(page_height_new, gap)
        context_bottom_box = (0, 0, page_width_new, context_bottom_end)
        context_bottom_text_add = pages[current_page + 1].crop(context_bottom_box).extract_text() if current_page < len(pages) - 1 else ""
        context_bottom_text = context_bottom_text + context_bottom_text_add

    page_num = current_page + 1
    return {
        'page_num': page_num,
        'y_position': y_top,
        'keyword': start_keyword,
        'context_above': context_top_text[-200:],
        'context_below': context_bottom_text[:500],
    }

def is_table_title(context):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    system_prompt = f"""
    你只需要判断关键词所在位置是不是对应的表格的开头。

    请严格按照以下规则工作：
    1.理解关键词所在的上下文，判断它是否是表格的标题
    2.判断关键词所在位置是否在表格的上方，如果在表格的下方或者表格中间，则不是标题
    3.如果关键词所在位置在表格的上方，并且关键词下方表格内容与关键词相关，则可以判断它是表格的标题
    """
    user_prompt = f"""
    ## 关键词
    {context['keyword']}

    ## 关键词所在的上下文
    {context['context_above']}
    {context['context_below']}

    ## 任务
    请分析关键词及其所在上下文，判断关键词所在的位置是否是它对应的表格的标题。

    ## 输出格式要求
    请严格按照以下 JSON 格式返回结果：
    {{
        "is_title":true,
        "extraction":"说明提取情况，例如：关键词在表格上方，且下方表格内容与关键词相关，因此判断为标题。"
    }}

    注意：
    1. 只能返回纯 JSON 格式，不能包含任何其他文本
    2. is_title 字段必须是布尔值 true 或 false，表示是否是标题
    3. extraction 字段必须包含提取情况的说明
    """
    completion = client.chat.completions.create(
        model=AI_MODEL_TITLE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print("🤖 正在调用 AI API 进行判断...")
    result = completion.choices[0].message.content
    print("判断结果：", result)
    if result.startswith("```json"):
        result = result.strip("```json")
    if result.endswith("```"):
        result = result.strip("```")
    try:
        result_dict = json.loads(result)
    except:
        return False
    return result_dict.get("is_title", False)

def locate_markers(pdf_path, start_keyword, end_keyword, mid_word):
    print(f"🔍 开始扫描文件：{pdf_path}")
    print(f"   寻找起始标记：'{start_keyword}'")
    print(f"   寻找结束标记：'{end_keyword}'")
    print("-" * 60)
    start_info = None
    end_info = None
    mid_key = None
    with pdfplumber.open(pdf_path) as pdf:
        if mid_word:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text()
                if not text:
                    continue
                start_words = re.findall(start_keyword, text)
                if start_words:
                    print(f"第{page_num}页匹配到关键词：{start_words}")
                for word in start_words:
                    if start_info is None:
                        matches = page.search(word)
                        if matches:
                            bbox = matches[0]
                            start_info = {
                                'page': page_num,
                                'y_top': bbox['top'],
                                'y_bottom': bbox['bottom'],
                                'text_preview': word
                            }
                            context = extract_context(pdf.pages, i, bbox, word)
                            is_title = is_table_title(context)
                            if is_title == False:
                                print(f"第{page_num}页的'{word}'位置可能不是表格标题，继续寻找其他匹配项...")
                                start_info = None
                            else:
                                print(f"✅ [第 {page_num} 页] 找到起始标记!")
                                print(f"   垂直位置 (Top): {bbox['top']:.2f}")
                                print(f"   垂直位置 (Bottom): {bbox['bottom']:.2f}")
                                mid_key = re.findall(mid_word, text)[0]
                if end_keyword and start_info:
                    word = "合并" + mid_key + end_keyword
                    if start_info and end_info is None and word:
                        print(f"第{page_num}页正在寻找结束标记：'{word}'")
                        matches = page.search(word)
                        if matches:
                            bbox = matches[0]
                            end_info = {
                                'page': page_num,
                                'y_top': bbox['top'],
                                'y_bottom': bbox['bottom'],
                                'text_preview': word
                            }
                            print(f"✅ [第 {page_num} 页] 找到结束标记!")
                            print(f"   垂直位置 (Top): {bbox['top']:.2f}")
                            print(f"   垂直位置 (Bottom): {bbox['bottom']:.2f}")
                elif end_keyword is None:
                    if start_info and end_info is None:
                        end_info = {
                            'page': len(pdf.pages),
                            'y_top': pdf.pages[-1].height,
                            'y_bottom': pdf.pages[-1].height,
                            'text_preview': '文档末尾'
                        }
                        print(f"⚠️  警告：未找到结束标记 '{end_keyword}'，将提取到文档末尾。")
                if start_info and end_info:
                    break
        else:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text()
                if not text:
                    continue
                if start_info is None:
                    matches = page.search(start_keyword)
                    if matches:
                        bbox = matches[0]
                        start_info = {
                            'page': page_num,
                            'y_top': bbox['top'],
                            'y_bottom': bbox['bottom'],
                            'text_preview': start_keyword
                        }
                        context = extract_context(pdf.pages, i, bbox, start_keyword)
                        is_title = is_table_title(context)
                        if is_title == False:
                            print(f"第{page_num}页的'{start_keyword}'位置可能不是表格标题，继续寻找其他匹配项...")
                            start_info = None
                        else:
                            print(f"✅ [第 {page_num} 页] 找到起始标记!")
                            print(f"   垂直位置 (Top): {bbox['top']:.2f}")
                            print(f"   垂直位置 (Bottom): {bbox['bottom']:.2f}")
                if start_info and end_info is None and end_keyword:
                    matches = page.search(end_keyword)
                    if matches:
                        bbox = matches[0]
                        end_info = {
                            'page': page_num,
                            'y_top': bbox['top'],
                            'y_bottom': bbox['bottom'],
                            'text_preview': end_keyword
                        }
                        print(f"✅ [第 {page_num} 页] 找到结束标记!")
                        print(f"   垂直位置 (Top): {bbox['top']:.2f}")
                        print(f"   垂直位置 (Bottom): {bbox['bottom']:.2f}")
                if start_info and end_info:
                    break
    print("-" * 60)
    if not start_info:
        print(f"❌ 错误：未找到起始标记 '{start_keyword}'")
        start_info = {
            'page': 1,
            'y_top': 0,
            'y_bottom': 0,
            'text_preview': 'no_find'
        }
    else:
        print(f"📍 起始位置确认：第 {start_info['page']} 页, Y={start_info['y_top']:.2f}")
    if not end_info:
        print(f"⚠️  警告：未找到结束标记 '{end_keyword}'")
        with pdfplumber.open(pdf_path) as pdf:
            end_info = {
                'page': len(pdf.pages),
                'y_top': pdf.pages[-1].height,
                'y_bottom': pdf.pages[-1].height,
                'text_preview': '文档末尾'
            }
    else:
        print(f"📍 结束位置确认：第 {end_info['page']} 页, Y={end_info['y_top']:.2f}")
    return start_info, end_info

def extract_tables_in_range(pdf_path, start_page, start_y, end_page, end_y):
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page):
            page_num = i + 1
            page = pdf.pages[i]
            page_height = page.height
            page_width = page.width
            crop_box = None
            if page_num == start_page and page_num == end_page:
                crop_box = (0, start_y, page_width, end_y)
                print(f"📄 第 {page_num} 页: 裁剪区域 Y[{start_y:.2f} - {end_y:.2f}]")
            elif page_num == start_page:
                crop_box = (0, start_y, page_width, page_height)
                print(f"📄 第 {page_num} 页: 裁剪区域 Y[{start_y:.2f} - 底部]")
            elif page_num == end_page:
                crop_box = (0, 0, page_width, end_y)
                print(f"📄 第 {page_num} 页: 裁剪区域 Y[顶部 - {end_y:.2f}]")
            elif start_page < page_num < end_page:
                crop_box = (0, 0, page_width, page_height)
                print(f"📄 第 {page_num} 页: 整页提取")
            if crop_box:
                cropped_page = page.crop(crop_box)
                tables = cropped_page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        clean_table = []
                        for row in table:
                            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            if any(clean_row):
                                clean_table.append(clean_row)
                        if len(clean_table) > 1:
                            all_tables.append({
                                'page': page_num,
                                'table': clean_table,
                                'rows': len(clean_table)
                            })
                            print(f"   ✅ 提取到表格: {len(clean_table)} 行")
    return all_tables

def merge_and_save(tables):
    if not tables:
        print("❌ 没有可保存的表格")
        return None
    print(f"\n📊 共提取到 {len(tables)} 个表格片段")
    all_data = []
    header = None
    for item in tables:
        table = item['table']
        if not table:
            continue
        current_header = table[0]
        data_rows = table[1:]
        if header is None:
            header = current_header
            all_data.extend(data_rows)
            print(f"   设定表头 (来自第 {item['page']} 页)")
        else:
            if current_header == header:
                all_data.extend(data_rows)
                print(f"   合并续表 (来自第 {item['page']} 页, 跳过重复表头)")
            else:
                all_data.extend(table)
                print(f"   合并新数据 (来自第 {item['page']} 页)")
    if header and all_data:
        max_cols = len(header)
        for i, row in enumerate(all_data):
            if len(row) < max_cols:
                all_data[i] = row + [""] * (max_cols - len(row))
            elif len(row) > max_cols:
                all_data[i] = row[:max_cols]
        df = pd.DataFrame(all_data, columns=header)
        print(f"   总行数: {len(df)}")
        print(f"   总列数: {len(df.columns)}")
        print("\n📋 数据预览 :")
        print(df.to_string())
        return df
    else:
        print("❌ 未能构建有效数据")
        return None

def extract_non_table_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return ""
        first_page = pdf.pages[0]
        table_finder = first_page.find_tables({
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "edge_min_length": 20,
            "snap_tolerance": 3
        })
        table_bboxes = [table.bbox for table in table_finder]
        non_table_text = []
        for text_block in first_page.chars:
            x, y = text_block['x0'], text_block['top']
            in_table = any(
                bbox[0] - 1 <= x <= bbox[2] + 1 and
                bbox[1] - 1 <= y <= bbox[3] + 1
                for bbox in table_bboxes
            )
            if not in_table:
                non_table_text.append(text_block['text'])
        cleaned_text = re.sub(r'\n\s*\n', '\n', ''.join(non_table_text))
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        return cleaned_text

def extract_key_metrics_text(pdf_file, max_pages=15):
    """
    提取年报'主要会计数据/主要财务指标'章节文本。
    该章节集中披露加权平均净资产收益率、扣非净利润、基本每股收益等核心指标，
    供 LLM 直接取用，避免现算导致口径偏差。
    """
    try:
        parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or ""
                if ("主要会计数据" in text) or ("主要财务指标" in text):
                    parts.append(f"--- 第 {page.page_number} 页 ---\n{text}")
        return "\n".join(parts)
    except Exception as e:
        print(f"⚠️ 提取主要会计数据章节失败: {e}")
        return ""

def safe_json_parse(s, default=None):
    if not isinstance(s, str):
        return default
    s = s.strip()
    if not s:
        return default
    if s.startswith('\ufeff') or s.startswith('\xef\xbb\xbf'):
        s = s[1:] if s.startswith('\ufeff') else s[3:]
    try:
        return json.loads(s)
    except:
        return default

def process_table(table_type, pdf_type, pdf_file, rules, stock_code=None, stock_abbr=None):
    # 完全原样保留 pdf2csv 中的 process_table 函数
    raw_table = []
    for i, type in enumerate(table_type):
        if pdf_type == '摘要':
            print('这是报告摘要')
            return "摘要", None
        elif pdf_type == '半年度报告' or pdf_type == '年度报告':
            marker_start = f"合并{type}"
            marker_end = f"母公司{type}"
            mid_word = None
        elif pdf_type == '第三季度报告':
            marker_start = f"合并.*?{type}"
            marker_end = f"{table_type[i + 1]}" if i + 1 < len(table_type) else None
            mid_word = f"合并(.*?){type}"
        else:
            marker_start = f"合并.*?{type}"
            marker_end = f"{table_type[i + 1]}" if i + 1 < len(table_type) else None
            mid_word = f"合并(.*?){type}"

        start_pos, end_pos = locate_markers(pdf_file, marker_start, marker_end, mid_word)
        if start_pos['text_preview'] == 'no_find':
            with pdfplumber.open(pdf_file) as pdf:
                start_pos = {
                    'page': 1,
                    'y_top': 0,
                    'y_bottom': 0,
                    'text_preview': '文档开头'
                }
                end_pos = {
                    'page': len(pdf.pages),
                    'y_top': pdf.pages[-1].height,
                    'y_bottom': pdf.pages[-1].height,
                    'text_preview': '文档末尾'
                }
        if end_pos is None:
            print("未找到结束标志，应将下一个开始标志作为结束标志")
            _, end_pos = locate_markers(pdf_file, f"合并.*?{table_type[i + 1]}" if i + 1 < len(table_type) else None, marker_end,
                                         f"合并(.*?){table_type[i + 1]}" if i + 1 < len(table_type) else None)
        START_PAGE = start_pos['page']
        START_Y = start_pos['y_top']
        END_PAGE = end_pos['page']
        END_Y = end_pos['y_top']

        print("=" * 60)
        print(f"🚀 开始提取合并{type}表格")
        print("=" * 60)

        tables = extract_tables_in_range(pdf_file, START_PAGE, START_Y, END_PAGE, END_Y)
        if tables:
            for idx, item in enumerate(tables):
                tables[idx]['table'] = clean_table_data(tables[idx]['table'])
        if tables:
            df = merge_and_save(tables)
            raw_table.append(df)
        else:
            print("❌ 未提取到任何表格")

    non_table_text = extract_non_table_text(pdf_file)
    key_metrics_text = extract_key_metrics_text(pdf_file)
    if key_metrics_text:
        non_table_text = non_table_text + "\n\n## 主要会计数据/主要财务指标（研报披露的核心指标，优先直接取用，注意区分本报告期与上年同期列）\n" + key_metrics_text

    all_tables = []
    report_year = None
    for i in range(len(rules)):
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        # 以下提示词完全原样保留
        system_prompt = f"""
        你是一位专业的财务数据提取专家。你的任务是从财务报表 DataFrame 中，
        根据字段定义表的字段类型和字段说明，准确提取对应的数据值。

        请严格按照以下规则工作：
        1. 理解每个字段的中文名称和字段说明
        2. 在 DataFrame 的列名和行内容中查找匹配的数据，优先取用研报已披露的现成指标值
        3. 注意数值单位（万元、元、%等）， DataFrame 财务报表数据以元为单位，必须统一转换为字段定义中的单位，并将转换过程记录在 extraction_notes 中，例如经营活动产生的现金流量净额为-57721288.63元，字段定义表中单位为万元，则需要转换为-5772.13万元，并记录为：现金流量净额-57721288.63元，除以10000转换为-5772.13万元
        4. 只能从 DataFrame 和报告文本中提取数据，不允许使用任何外部知识或数据，更不要猜测
        5. 对于无法直接找到的数据，可以尝试根据字段定义表的字段说明和计算公式进行计算（若字段定义表的字段说明和计算公式有冲突，则以计算公式为准），若无法计算，返回 null
        6. 若输出结果存在 null 值，需要重新检查是否有遗漏的字段或计算错误，确保所有字段都被正确提取或计算
        7. 再次检查单位转换是否正确，通过将提取的数值乘以单位转换后的倍数，验证是否与原始数据一致，确保没有单位转换错误
        8. 净资产收益率类字段（roe、roe_weighted_excl_non_recurring）必须优先从"主要会计数据/主要财务指标"章节取用研报披露的现成数值（取本报告期当年列）：roe 对应"加权平均净资产收益率"，roe_weighted_excl_non_recurring 对应"扣除非经常性损益后的加权平均净资产收益率"；仅当研报确实未披露该指标时，才允许按下方公式计算，且必须在 extraction_notes 中注明"采用计算值"及计算过程

        计算公式：
        ROE = 净利润 / [(期初所有者权益合计 + 期末所有者权益合计) / 2]（仅当研报未披露加权平均净资产收益率时使用）
        每股净资产 = 所有者权益合计 / 总股本（或实收资本）
        销售毛利率 = (营业总收入−营业成本) / 营业总收入 * 100%
        销售净利率 = 净利润 / 营业总收入 * 100%
        同比增长率 = (本期数值 - 上期数值) / 上期数值 * 100%
        每股经营现金流量 = 经营活动产生的现金流量净额 / 总股本（或实收资本）
        所有者权益合计 = 总资产 - 总负债
        """

        user_prompt = f"""
        ## 字段定义表
        {rules[i].to_markdown(index=False)}

        ## DataFrame 财务报表数据
        {raw_table[0].to_markdown(index=False) if len(raw_table) > 0 else ''}
        {raw_table[1].to_markdown(index=False) if len(raw_table) > 1 else ''}
        {raw_table[2].to_markdown(index=False) if len(raw_table) > 2 else ''}

        ## 报告第一页内容
        {non_table_text}
        深交所股票简称与代码对照表：{NAME2CODE}
        上交所股票代码与简称对照表：{CODE2NAME}

        ## 已知信息
        - 股票代码：{stock_code if stock_code else "待识别"}
        - 股票简称：{stock_abbr if stock_abbr else "待识别"}

        ## 任务
        请分析 DataFrame 数据和报告第一页内容，提取上述字段定义中的所有字段值，按照字段定义的顺序返回，返回类型为“字段类型”。

        ## 输出格式要求
        请输出严格的 JSON 格式，结构如下：
        {{
            "serial_number": null,
            "stock_code": "股票代码字符串",
            "stock_abbr": "股票简称字符串",
            "asset_cash_and_cash_equivalents": 12345.67,
            "asset_accounts_receivable": 23456.78,
            ...
            "report_period": "FY",
            "report_year": 1970,
            "extraction_notes": "提取过程中的说明，如哪些字段未找到、未找到的字段是否能通过计算得到、单位转换说明等"
        }}

        注意：
        1. 数值字段直接返回数字，不要带单位
        2. 字符串字段返回字符串
        3. 找不到的字段返回 null
        4. 必须包含 extraction_notes 说明提取情况，其中必须使用中文字符，包括标点符号
        4. 只能返回纯 JSON 格式，不能包含任何其他文本
        5. 字段名称必须与字段定义表中的字段名称完全一致
        6. 返回的字典必须包含字段定义表中的所有字段，缺失字段必须以 null 填充
        7. 字段必须按照字段定义表中字段的顺序返回
        8. '净利润'类字段一律取"归属于母公司股东的净利润"（优先采用"主要会计数据/主要财务指标"章节披露的归母净利润，取本报告期当年列）；利润表中的"净利润"含少数股东损益，禁止直接用于净利润类字段的提取和计算
        9. 任意同比的计算公式都是：同比=（本期数值-上期数值）/上期数值*100%
        10. DataFrame 财务报表数据中的所有数据都是以元为单位的原始数据，字段定义表中的单位可能是万元或元，请根据字段定义表中的单位进行必要的转换
        11. 必须先统一单位，再进行计算
        """

        completion = client.chat.completions.create(
            model=AI_MODEL_EXTRACT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=5000
        )
        print("🤖 正在调用 AI API 进行字段提取...")
        result = completion.choices[0].message.content
        print(result)
        if result.startswith("```json"):
            result = result.strip("```json")
        if result.endswith("```"):
            result = result.strip("```")
        result_dict = safe_json_parse(result, default={})
        if i == 0:
            report_year = result_dict.get('report_year')
        if i < 3:
            all_tables.append({table_type[i]: result_dict})
        else:
            all_tables.append({'核心业绩指标表': result_dict})

    return all_tables, report_year

def process_single_pdf(full_path, filename, table_type, rules, output_csvs, db_config, exchange='sz'):
    """
    处理单个PDF的完整流程（包含重试），返回处理结果状态。
    该函数由线程池调用。
    exchange: 'sz'=深交所（文件名：公司名：报告.pdf），'sh'=上交所（文件名：代码_日期_随机码.pdf）
    """
    if exchange == 'sh':
        # 上交所：文件名格式 600080_20230428_FQ2V.pdf
        stock_code = filename[:6]
        stock_name = CODE2NAME.get(stock_code, '')
    else:
        # 深交所：文件名格式 万邦德：2022年年度报告.pdf
        stock_name = filename.split('：')[0].replace(' ', '')  # 九 芝 堂 等文件名含空格，去空格后再查表
        stock_code = NAME2CODE.get(stock_name, '')
    
    print(f"\n{'='*60}\n处理文件: {filename} (股票: {stock_code} {stock_name})\n{'='*60}")

    # 识别 PDF 类型并跳过摘要
    pdf_type = None
    try:
        with pdfplumber.open(full_path) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
            if '摘要' in first_page_text:
                print(f"⚠️ {filename} 检测到报告摘要，跳过处理")
                return {'filename': filename, 'status': 'skipped', 'reason': '摘要'}
            for t in ['半年度报告', '年度报告', '第三季度报告', '第一季度报告']:
                if t in first_page_text:
                    pdf_type = t
                    break
    except Exception as e:
        print(f"❌ {filename} 打开PDF失败: {e}")
        return {'filename': filename, 'status': 'error', 'reason': str(e)}

    if pdf_type is None:
        print(f"⚠️ {filename} 无法识别报告类型，默认按年度报告处理")
        pdf_type = '年度报告'

    # 每个线程使用独立的校验器实例（内部会创建独立的数据库连接）
    validator = SingleRecordValidator(db_config)

    success = False
    last_errors = []
    for attempt in range(1, 4):
        print(f"\n🔄 {filename} 第 {attempt} 次尝试...")
        try:
            results, report_year = process_table(table_type, pdf_type, full_path, rules, stock_code, stock_name)
        except Exception as e:
            print(f"⚠️ {filename} 第 {attempt} 次尝试异常: {e}")
            time.sleep(5 * attempt)
            continue
        if results == "摘要":
            return {'filename': filename, 'status': 'skipped', 'reason': '摘要'}
        if results is None:
            print(f"{filename} 处理失败，结果为空")
            continue

        # 提取四个字典
        bal_dict = results[0].get('资产负债表', {})
        inc_dict = results[1].get('利润表', {})
        cash_dict = results[2].get('现金流量表', {})
        core_dict = results[3].get('核心业绩指标表', {})

        # 补充通用字段
        for d in [bal_dict, inc_dict, cash_dict, core_dict]:
            d['stock_code'] = stock_code
            d['stock_abbr'] = stock_name
            if report_year:
                d['report_year'] = report_year

        # 校验并入库
        is_valid, errors = validator.validate_and_insert(core_dict, inc_dict, bal_dict, cash_dict)
        if is_valid:
            print(f"✅ {filename} 校验通过，已入库")
            # 使用文件锁安全写入CSV
            with csv_lock:
                pd.DataFrame([bal_dict]).to_csv(output_csvs['资产负债表'], mode='a', index=False, header=False)
                pd.DataFrame([inc_dict]).to_csv(output_csvs['利润表'], mode='a', index=False, header=False)
                pd.DataFrame([cash_dict]).to_csv(output_csvs['现金流量表'], mode='a', index=False, header=False)
                pd.DataFrame([core_dict]).to_csv(output_csvs['核心'], mode='a', index=False, header=False)
            success = True
            break
        else:
            last_errors = errors
            print(f"❌ {filename} 校验失败: {'; '.join(errors)}")

    if success:
        return {'filename': filename, 'status': 'success'}
    else:
        return {'filename': filename, 'status': 'failed', 'errors': last_errors}

# ================== 校验器（基于 check.py 改编为单行校验） ==================
class SingleRecordValidator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.errors = []

    def _get_connection(self):
        """创建数据库连接"""
        return pymysql.connect(**self.db_config)

    def validate_format(self, record_dict):
        errs = []
        if not re.match(r'^\d{6}$', str(record_dict.get('stock_code', ''))):
            errs.append("股票代码格式错误")
        if record_dict.get('report_period') not in ['FY', 'Q1', 'HY', 'Q3']:
            errs.append("报告期格式错误")
        return errs

    def validate_business_logic(self, core_dict, income_dict, balance_dict, cash_dict):
        errs = []
        # 核心表销售净利率
        try:
            if core_dict.get('total_operating_revenue') and core_dict.get('net_profit_10k_yuan'):
                calc = core_dict['net_profit_10k_yuan'] / core_dict['total_operating_revenue'] * 100
                if abs(calc - core_dict.get('net_profit_margin', 0)) > 1:
                    errs.append("核心表销售净利率计算不一致")
        except:
            pass
        # 利润表毛利率
        try:
            if income_dict.get('total_operating_revenue') and income_dict.get('operating_expense_cost_of_sales'):
                calc_margin = (income_dict['total_operating_revenue'] - income_dict['operating_expense_cost_of_sales']) / income_dict['total_operating_revenue'] * 100
                if abs(calc_margin - core_dict.get('gross_profit_margin', 0)) > 1:
                    errs.append("毛利率计算不一致")
        except:
            pass
        # 资产负债表平衡
        try:
            diff = balance_dict['asset_total_assets'] - (balance_dict['liability_total_liabilities'] + balance_dict['equity_total_equity'])
            if abs(diff) > 10:
                errs.append("资产负债表不平衡")
            calc_ratio = balance_dict['liability_total_liabilities'] / balance_dict['asset_total_assets'] * 100
            if abs(calc_ratio - balance_dict.get('asset_liability_ratio', 0)) > 0.1:
                errs.append("资产负债率计算不一致")
        except:
            pass
        # 现金流量表
        try:
            calc_cf = (cash_dict['operating_cf_net_amount'] + cash_dict['investing_cf_net_amount'] + cash_dict['financing_cf_net_amount']) * 10000
            if abs(cash_dict['net_cash_flow'] - calc_cf) > abs(calc_cf) * 0.1:
                errs.append("现金流量表三大现金流之和不等于净现金流")
        except:
            pass
        # 跨表一致性
        if income_dict.get('total_operating_revenue') != core_dict.get('total_operating_revenue'):
            errs.append("利润表与核心表营业收入不一致")
        if income_dict.get('net_profit') != core_dict.get('net_profit_10k_yuan'):
            errs.append("利润表与核心表净利润不一致")
        return errs

    def validate_all(self, core_dict, income_dict, balance_dict, cash_dict):
        self.errors = []
        for d in [core_dict, income_dict, balance_dict, cash_dict]:
            self.errors.extend(self.validate_format(d))
        self.errors.extend(self.validate_business_logic(core_dict, income_dict, balance_dict, cash_dict))
        return len(self.errors) == 0, self.errors

    def insert_tables_transaction(self, core_dict, income_dict, balance_dict, cash_dict):
        """
        在同一事务中插入四张表，全部成功则提交，否则回滚。
        返回 bool 表示是否成功。
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # 1. 核心业绩指标表
            self._insert_one_table(cursor, 'core_performance_indicators_sheet', core_dict)
            # 2. 利润表
            self._insert_one_table(cursor, 'income_sheet', income_dict)
            # 3. 资产负债表
            self._insert_one_table(cursor, 'balance_sheet', balance_dict)
            # 4. 现金流量表
            self._insert_one_table(cursor, 'cash_flow_sheet', cash_dict)
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"事务执行失败，已回滚: {e}")
            raise  # 将异常继续抛出，便于上层捕获
        finally:
            conn.close()

    def _insert_one_table(self, cursor, table_name, record_dict):
        """单表插入（使用已有的cursor）"""
        exclude_fields = {'extraction_notes', 'serial_number', 'exchange_rate'}
        filtered_dict = {k: v for k, v in record_dict.items() if k not in exclude_fields}
        # 转换为DataFrame便于处理None
        df = pd.DataFrame([filtered_dict])
        df = df.replace(np.nan, None)
        columns = list(df.columns)
        placeholders = ", ".join(["%s"] * len(columns))
        update_clause = ", ".join([f"{col} = VALUES({col})" for col in columns])
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}"
        values = tuple(df.iloc[0][col] for col in columns)
        cursor.execute(sql, values)

    # ---------- 修改：验证和插入流程（将校验和入库分离，但仍组合） ----------
    def validate_and_insert(self, core_dict, income_dict, balance_dict, cash_dict):
        """
        执行校验，若通过则在一个事务中插入四张表，并记录日志。
        注意：日志表写入在插入操作之后，且使用独立连接，不影响事务。
        """
        # 1. 校验
        is_valid, errors = self.validate_all(core_dict, income_dict, balance_dict, cash_dict)
        status = 1 if is_valid else 0

        # 2. 记录日志（使用独立连接，即时提交）
        self._log_validation_independent('all_tables', 1, 1 if is_valid else 0, 0 if is_valid else 1, status, '; '.join(errors))

        if not is_valid:
            return False, errors

        # 3. 校验通过，执行事务插入
        try:
            self.insert_tables_transaction(core_dict, income_dict, balance_dict, cash_dict)
            return True, []
        except Exception as e:
            # 插入失败，记录错误并返回
            error_msg = f"数据库插入失败: {str(e)}"
            self._log_validation_independent('all_tables', 1, 0, 1, 0, error_msg)
            return False, [error_msg]

    def _log_validation_independent(self, table_name, total_records, passed, failed, status, error_details):
        """独立的日志记录方法，每次调用都新建连接并提交，不影响主事务"""
        conn = pymysql.connect(**self.db_config)
        try:
            cursor = conn.cursor()
            batch_id = str(uuid.uuid4())
            sql = """
            INSERT INTO log_data_validation 
            (validation_batch_id, table_name, validation_type, total_records, 
             passed_records, failed_records, validation_status, error_details, validated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (batch_id, table_name, 'SINGLE', total_records, passed, failed, status, error_details, datetime.now()))
            conn.commit()
        except Exception as e:
            print(f"日志写入失败: {e}")
        finally:
            conn.close()

# ================== 主流程 ==================
def main():
    # 修复：重定向/文件输出时强制 UTF-8，避免 emoji 打印在 GBK 编码下抛 UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    table_type = ['资产负债表', '利润表', '现金流量表']
    rules = []
    for t in table_type:
        rules.append(pd.read_excel(FIELDS_XLSX, sheet_name=t))
    rules.append(pd.read_excel(FIELDS_XLSX, sheet_name='核心业绩指标表'))

    output_csvs = {
        '资产负债表': '资产负债表.csv',
        '利润表': '利润表.csv',
        '现金流量表': '现金流量表.csv',
        '核心': '核心业绩指标表.csv'
    }
    # 初始化 CSV 文件（写入表头）
    for path in output_csvs.values():
        if not os.path.exists(path):
            pd.DataFrame().to_csv(path, index=False)

    # 构建任务参数列表
    tasks = []
    total_files = 0
    for exchange, pdf_dir in PDF_DIRS:
        files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        total_files += len(files)
        print(f"目录 {pdf_dir} 共发现 {len(files)} 个PDF文件（{exchange}）")
        for filename in files:
            full_path = os.path.join(pdf_dir, filename)
            tasks.append((full_path, filename, table_type, rules, output_csvs, DB_CONFIG, exchange))
    print(f"共发现 {total_files} 个PDF文件，将使用最多5个线程并行处理。")

    # 使用线程池并行处理
    max_workers = 5
    results_summary = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_pdf, *task): task[1]
            for task in tasks
        }
        # 处理完成的任务
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                results_summary.append(result)
                # 实时打印进度
                if result['status'] == 'success':
                    print(f"🏁 {filename} 处理成功")
                elif result['status'] == 'skipped':
                    print(f"⏭️ {filename} 跳过: {result.get('reason', '')}")
                else:
                    print(f"⚠️ {filename} 最终失败: {result.get('errors', [])}")
            except Exception as e:
                print(f"🔥 {filename} 任务执行异常: {e}")
                results_summary.append({'filename': filename, 'status': 'error', 'reason': str(e)})

    # 打印最终统计
    print("\n" + "="*60)
    print("所有任务处理完毕，统计结果：")
    success_count = sum(1 for r in results_summary if r['status'] == 'success')
    skip_count = sum(1 for r in results_summary if r['status'] == 'skipped')
    fail_count = sum(1 for r in results_summary if r['status'] == 'failed')
    error_count = sum(1 for r in results_summary if r['status'] == 'error')
    print(f"成功: {success_count}, 跳过: {skip_count}, 校验失败: {fail_count}, 异常: {error_count}")

    # 记录失败和异常的文件
    failed_list = [r for r in results_summary if r['status'] in ('failed', 'error')]
    if failed_list:
        with open('failed_files.txt', 'w', encoding='utf-8') as f:
            f.write("以下文件处理失败或异常，建议重试：\n")
            for item in failed_list:
                reason = item.get('errors', item.get('reason', ''))
                if isinstance(reason, list):
                    reason = '; '.join(reason)
                f.write(f"{item['filename']}  -- 状态: {item['status']} -- 原因: {reason}\n")
        print(f"失败及异常文件列表已保存至 failed_files_shang.txt，共 {len(failed_list)} 个。")
    else:
        print("所有文件均成功或跳过，无失败记录。")

if __name__ == "__main__":
    main()
    
