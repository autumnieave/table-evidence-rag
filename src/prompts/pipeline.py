"""业务级 Prompt —— 多轮澄清字段提取 / 摘要 / 图片检测

原散落在 pipelines/rag_pipeline.py 与 utils/helpers.py（各存一份，存在缩进/转义漂移），
2026-08-22 统一收口到本模块；模板以 RAGPipeline 实际运行路径为准。
"""

# 模块级版本号：修改本文件任一 Prompt 文本后递增（prompts/registry.json 同步登记）
PIPELINE_PROMPT_VERSION = "2026-09-09-v1"

# 多轮澄清：从用户问题中提取过滤字段（pipelines.rag_pipeline._parse_filters_with_llm 使用）
FILTER_EXTRACT_PROMPT_TEMPLATE = """你是一个金融查询解析助手。请结合对话历史（若有），从用户最新问题中提取以下字段，并以 JSON 格式返回。如果某个字段未提及，则其值为 null。
    
    {history_section}
        
    今日日期: {today}
    当前年份: {current_year}

    字段说明：
    - stock_name: 股票简称，例如 "马应龙"
    - stock_code: 股票代码，例如 "600993"
    - start_date: 查询开始日期，表示研报的发布日期（对应数据库字段 publishDate ）应在此日期之后，格式 YYYY-MM-DD。**仅当明确用户要求查询特定日期的研报，如"最近的研报"、"近三个月的研报"、"2024年的研报"等与研报发布时间相关的条件时才填充**。
    - end_date: 查询结束日期，表示研报的发布日期（对应数据库字段 publishDate ）应在此日期之前，格式 YYYY-MM-DD。**仅当明确用户要求查询特定日期的研报，如"最近的研报"、"近三个月的研报"、"2024年的研报"等与研报发布时间相关的条件时才填充**。
    - rating: 评级，例如 "买入"、"增持"
    - org_name: 券商全称或简称，例如 "信达证券"
    - researcher: 研究员姓名，例如 "唐爱金"
    - industry: 所属行业，例如 "中药"
    - title: 研报标题，例如 "云南白药2025半年报业绩点评：医药工业双位数增长，经营质量稳步提升"
    - doc_type: 研报类型，若用户明确询问"个股"、"公司"填 "stock"；询问"行业"、"板块"填 "industry"；否则为 null。

    用户最新问题：{question}
    
    注意：
    1. 所有日期范围（如“近三个月”、“上周”、“今年”）都应基于今日日期（{today}）进行精确计算。例如，“近三个月”应计算为从 {today} 往前推三个月的那一天。
    2. 若用户使用代词（如“它”、“该公司”），请根据对话历史中的股票信息填充。
    请仅返回 JSON 格式，不要包含任何其他文字。
    JSON:"""


# 摘要生成（RAGPipeline._generate_summaries 与 utils.helpers._generate_summaries 共用）
SUMMARY_PROMPT_TEMPLATE = """请用一句话概括以下金融研报片段的核心内容，不超过50个字，只输出概括文本，不要添加任何前缀或解释。

    片段内容：
    {text}

    概括："""


# 图片标题检测（RAGPipeline._extract_image_title_with_llm 与 utils.helpers 同函数共用）
# 注：原 pipelines 版本在 {text} 后误带注释文本“# 限制长度，避免 token 过多”，统一时已修正
IMAGE_DETECT_PROMPT_TEMPLATE = """判断以下金融研报文本块中是否包含图片。如果包含，请提取图片的标题（通常位于图片上方）。如果不存在图片，请直接返回"无图片"。只返回提取到的标题文本，不要添加任何解释或前缀。

    文本块：
    {text}

    注意：
    1. 图片通常以 Markdown 格式嵌入，如 ![](url)，标题可能在图片上方的文本中。"""
