import os
import base64
import json
import csv
import pdfplumber
from io import BytesIO
from pdf2image import convert_from_path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
load_dotenv()

# ================== 配置区域 ==================
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")   # 从 .env 读取 DashScope API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-max"                            # 视觉理解模型

PDF_FOLDERS = [
    "data_corpus/全部数据/正式数据/附件2：财务报告/reports-深交所",
    "data_corpus/全部数据/正式数据/附件2：财务报告/reports-上交所",
]                                                                               # 存放所有PDF的文件夹（正式数据）
OUTPUT_CSV = "extracted_missing_fields.csv"      # 输出结果文件名
MAX_WORKERS = 3                                  # 并行线程数（避免API限流）
SKIP_EMPTY_FIELDS = False                        # 是否跳过三个字段全为null的结果（默认写入）



# 三个待提取字段的描述
TARGET_FIELDS_DESC = """
1. net_profit_excl_non_recurring：扣非净利润（万元）。即剔除非经常性损益后的净利润，单位需转换为万元。
2. net_profit_excl_non_recurring_yoy：扣非净利润同比增长（%）。计算方式：(本期扣非净利润 - 上年同期扣非净利润) / |上年同期扣非净利润| × 100%
3. roe_weighted_excl_non_recurring：加权平均净资产收益率（扣非）（%）。计算公式为：扣非净利润 / 加权平均净资产 × 100%。
"""

# CSV写入锁，防止多线程写冲突
csv_lock = threading.Lock()
# ==============================================

def should_process(pdf_path):
    """
    判断PDF是否需要处理。
    返回 (process: bool, report_type: str)
    跳过规则：同时满足包含“年度报告”或“半年度报告”且不含“摘要”，即完整年报/半年报。
    其余一律处理。
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return False, "空文件"
            first_page_text = pdf.pages[0].extract_text() or ""
            # 识别具体报告类型
            if "第一季度报告" in first_page_text:
                return True, "Q1报告"
            elif "第三季度报告" in first_page_text:
                return True, "Q3报告"
            elif "半年度报告摘要" in first_page_text:
                return True, "半年报摘要"
            elif "年度报告摘要" in first_page_text:
                return True, "年报摘要"
            elif "半年度报告" in first_page_text and "摘要" not in first_page_text:
                return False, "完整半年报"
            elif "年度报告" in first_page_text and "摘要" not in first_page_text:
                return False, "完整年报"
            else:
                # 未明确识别，默认处理（可能是命名不规范，让模型判断）
                return True, "未知类型"
    except Exception as e:
        print(f"⚠️ 读取PDF失败 {pdf_path}: {e}")
        return False, "读取错误"

def pdf_to_base64_images(pdf_path, dpi=150):
    """将PDF每一页转为Base64图片（摘要/季报页数少，完整报告已被跳过）"""
    poppler_path = r"D:\poppler-25.12.0\Library\bin"
    images = convert_from_path(pdf_path, dpi=dpi, fmt='jpeg', poppler_path=poppler_path)
    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_images.append(b64)
    return base64_images

def extract_fields(pdf_path, stock_code=None, stock_abbr=None):
    """调用 Qwen VL 提取三个指定字段"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    base64_images = pdf_to_base64_images(pdf_path)

    content_parts = [
        {
            "type": "text",
            "text": f"""
你是一位专业财务分析师。现在给你一份上市公司的财务报告（第一季度报告/第三季度报告/半年度报告摘要/年度报告摘要之一），请从中提取或计算指定财务指标。

已知信息：
- 股票代码：{stock_code if stock_code else '请从报告中识别'}
- 股票简称：{stock_abbr if stock_abbr else '请从报告中识别'}

需要提取的字段及说明：
{TARGET_FIELDS_DESC}

提取规则：
- 优先在报告表格中寻找现成的“扣非净利润”、“扣非净利润同比增长率”、“加权平均净资产收益率（扣非）”等数值。
- 若找不到现成指标，则利用报告中列出的原始数据进行计算：
    - 扣非净利润：通常报告会直接披露，注意单位转换（元转为万元需除以10000）。
    - 扣非净利润同比增长率：若同时披露了本期和上年同期扣非净利润，请自行计算。
    - 加权平均净资产收益率（扣非）：若披露了期初净资产、期末净资产和扣非净利润，按公式计算。
- 如果报告中确实无法找到必要数据（如缺上期对比数据），对应字段填 null。

请以严格的 JSON 格式返回结果，字段必须完整：
{{
    "stock_code": "从报告中识别的股票代码",
    "stock_abbr": "股票简称",
    "report_year": 年份整数,
    "report_period": "Q1 / Q3 / HY / FY 之一",
    "net_profit_excl_non_recurring": 数值或null,
    "net_profit_excl_non_recurring_yoy": 数值或null,
    "roe_weighted_excl_non_recurring": 数值或null,
    "extraction_notes": "简要说明每个字段的数据来源及计算过程"
}}

只返回纯 JSON，不要包含任何 Markdown 标记或额外解释。
"""
        }
    ]

    for img_b64 in base64_images:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content_parts}],
        temperature=0.1,
        max_tokens=2000
    )

    result_text = response.choices[0].message.content.strip()
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]

    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return None

def process_one_pdf(pdf_path, filename):
    """处理单个PDF的完整流程，返回结果字典用于写入CSV"""
    process_flag, report_type = should_process(pdf_path)
    if not process_flag:
        print(f"⏭️ 跳过: {filename} ({report_type})")
        return {"filename": filename, "status": f"skipped_{report_type.replace(' ', '_')}"}

    print(f"🔄 处理: {filename} ({report_type})")
    try:
        # 可尝试从文件名解析简称（按实际格式调整）
        basename = os.path.splitext(filename)[0]
        if '：' in basename:
            stock_abbr = basename.split('：')[0]
            stock_code = None
        elif basename[:6].isdigit():
            stock_code = basename[:6]
            stock_abbr = None
        else:
            stock_abbr = None
            stock_code = None

        data = extract_fields(pdf_path, stock_code=stock_code, stock_abbr=stock_abbr)
        if data is None:
            print(f"❌ 提取失败: {filename} (JSON解析错误)")
            return {"filename": filename, "status": "extraction_failed"}

        data["filename"] = filename
        data["status"] = "success"
        # 可选：跳过全null结果
        if SKIP_EMPTY_FIELDS and all(data.get(f) is None for f in [
            "net_profit_excl_non_recurring",
            "net_profit_excl_non_recurring_yoy",
            "roe_weighted_excl_non_recurring"
        ]):
            data["status"] = "all_null"
            print(f"⚠️ 所有字段为null: {filename}")
        else:
            print(f"✅ 成功: {filename} -> {data.get('stock_code')} {data.get('report_year')}{data.get('report_period')}")
        return data

    except Exception as e:
        print(f"🔥 异常: {filename} - {e}")
        return {"filename": filename, "status": f"error: {e}"}

def write_csv_row(writer, row_dict, fieldnames):
    """线程安全地写入CSV一行"""
    with csv_lock:
        writer.writerow(row_dict)

def batch_process(pdf_folder, output_csv, max_workers=MAX_WORKERS, append_mode=False):
    """多线程批量处理

    Args:
        pdf_folder: PDF 文件夹路径
        output_csv: 输出 CSV 路径
        max_workers: 并行线程数
        append_mode: 是否追加写入（False 时先写表头）
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    print(f"📂 找到 {len(pdf_files)} 个 PDF 文件")

    fieldnames = [
        "filename", "stock_code", "stock_abbr", "report_year", "report_period",
        "net_profit_excl_non_recurring", "net_profit_excl_non_recurring_yoy",
        "roe_weighted_excl_non_recurring", "extraction_notes", "status"
    ]

    # 初始化CSV文件（写入表头；追加模式下跳过）
    if not append_mode:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_one_pdf, os.path.join(pdf_folder, f), f): f
            for f in pdf_files
        }

        # 打开CSV文件用于追加写入（注意模式为'a'）
        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    # 确保结果包含所有字段
                    row = {field: result.get(field) for field in fieldnames}
                    write_csv_row(writer, row, fieldnames)
                except Exception as e:
                    print(f"🔥 获取结果异常: {filename} - {e}")
                    row = {"filename": filename, "status": f"future_error: {e}"}
                    write_csv_row(writer, row, fieldnames)

    print(f"\n🎉 批量处理完成！结果已保存至: {output_csv}")

if __name__ == "__main__":
    for idx, folder in enumerate(PDF_FOLDERS):
        batch_process(folder, OUTPUT_CSV, MAX_WORKERS, append_mode=(idx > 0))
