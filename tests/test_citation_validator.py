"""pipelines.citation_validator L1 引用核验单元测试（临时语料，不依赖真实研报库）。"""

from pipelines.citation_validator import CitationValidator, _norm_key


def test_norm_key_normalizes():
    # 弯引号 / 双反斜杠 / 空白 归一化
    assert _norm_key("贵州茅台：2023年报.md") == _norm_key("贵州茅台：2023年报.md")
    assert _norm_key("a\\b\\c.md") == _norm_key("a/b/c.md")
    assert _norm_key("带 空 白 名.md") == _norm_key("带空白名.md")


def test_exact_locate(tmp_path):
    f = tmp_path / "贵州茅台2023年报.md"
    f.write_text("贵州茅台2023年营业收入 1500 亿元", encoding="utf-8")
    v = CitationValidator(corpus_root=str(tmp_path))
    status, located = v.locate(str(f))
    assert status == "exact" and located


def test_fuzzy_locate(tmp_path):
    (tmp_path / "贵州茅台2023年报.md").write_text("营收 1500 亿元", encoding="utf-8")
    v = CitationValidator(corpus_root=str(tmp_path))
    # 路径带反斜杠/引号差异，应模糊命中
    status, located = v.locate("测试数据\\贵州茅台2023年报.md")
    assert status == "fuzzy" and located


def test_relocate_by_directory_tail_not_basename_guess(tmp_path):
    # 外部数据集路径（B题数据及提交说明\...\附件5：研报数据\子目录\x.md）应
    # 按「语料根同名目录之后的尾段」重定位到本地语料根下，而不是靠文件名在全库索引里猜
    root = tmp_path / "附件5：研报数据"
    (root / "行业研报-解析结果-2.0").mkdir(parents=True)
    (root / "个股研报-解析结果-2.0").mkdir(parents=True)
    industry = root / "行业研报-解析结果-2.0" / "同名报告.md"
    stock = root / "个股研报-解析结果-2.0" / "同名报告.md"
    industry.write_text("行业口径 12.5 亿元", encoding="utf-8")
    stock.write_text("个股口径 99.9 亿元", encoding="utf-8")
    v = CitationValidator(corpus_root=str(root))
    external = "B题数据及提交说明\\全部数据\\正式数据\\附件5：研报数据\\行业研报-解析结果-2.0\\同名报告.md"
    status, located = v.locate(external)
    assert status == "fuzzy" and located
    assert located == str(industry.resolve())
    # 数字核验也应落到被引用的那一份文件，而不是同名的另一份
    rec = v.check_reference(external, "行业口径 12.5 亿元")
    assert rec["num_hit"] == rec["nums"] == 1
    wrong = v.check_reference(external, "个股口径 99.9 亿元")
    assert wrong["num_hit"] == 0


def test_relocate_rejects_nonexistent_tail(tmp_path):
    # 尾段在语料根下不存在时不得凭空命中（保持 missing）
    root = tmp_path / "语料根名"
    root.mkdir()
    (root / "a.md").write_text("营收 1 亿元", encoding="utf-8")
    v = CitationValidator(corpus_root=str(root))
    status, located = v.locate("外部\\语料根名\\子目录\\不存在.md")
    assert status == "missing" and located is None


def test_missing_locate(tmp_path):
    v = CitationValidator(corpus_root=str(tmp_path))
    status, located = v.locate("不存在.md")
    assert status == "missing" and located is None


def test_number_hit_and_summary(tmp_path):
    f = tmp_path / "金花股份2023年报.md"
    f.write_text("营业收入 2023 年达到 12.5 亿元，同比增长 30%", encoding="utf-8")
    v = CitationValidator(corpus_root=str(tmp_path))
    refs = [
        {"paper_path": str(f), "text": "营收 12.5 亿元，同比增长 30%"},
        {"paper_path": str(f), "text": "营收 999 亿元"},
        {"paper_path": "缺失文件.md", "text": "营收 12.5 亿元"},
    ]
    records = v.check_references(refs)
    summary = v.summarize(records)
    assert summary["total"] == 3
    assert summary["traceable"] == 2 and summary["missing"] == 1
    # 第一条含 12.5 与 30 两个数字，文件里都有 → 全命中
    assert records[0]["num_hit"] == records[0]["nums"]
    # 第二条 999 未命中
    assert records[1]["num_hit"] == 0 and "999" in records[1]["unhit"]
    # 缺失文件不参与数字核验
    assert records[2]["nums"] > 0 and records[2]["num_hit"] == 0


def test_latex_spaced_number_extract_and_match(tmp_path):
    # LaTeX 空格数字：8 7 . 5 应折叠为 87.5 并命中源文件
    f = tmp_path / '康美指数.md'
    f.write_text('区间价格涨幅达 $8 7 . 5 \\%$，指数从 1 200 点上涨', encoding='utf-8')
    v = CitationValidator(corpus_root=str(tmp_path))
    assert v.extract_numbers('涨幅达 $8 7 . 5\\%$') == ['87.5']
    records = v.check_references([{'paper_path': str(f), 'text': '涨幅 $8 7 . 5\\%$'}])
    assert records[0]['num_hit'] == records[0]['nums'] == 1


def test_number_in_text_unit_variants(tmp_path):
    # 单位换算：58.86 亿元 == 5,886 百万元
    f = tmp_path / '白云山.md'
    f.write_text('2025E 销售费用 5,886 百万元，营业总收入 79,001 百万元', encoding='utf-8')
    v = CitationValidator(corpus_root=str(tmp_path))
    haystack = v._normalize_for_match(f.read_text(encoding='utf-8'))
    assert v.number_in_text('58.86', haystack)[0] is False
    hit, unit_only = v.number_in_text('58.86', haystack, accept_unit_variants=True)
    assert hit and unit_only
    # 原生数字仍为精确命中，不计入 unit_only
    assert v.number_in_text('5886', haystack) == (True, False)


def test_check_reference_unit_variants(tmp_path):
    # check_reference 开启单位换算后记录 unit_hit，默认保持原行为
    f = tmp_path / '马应龙.md'
    f.write_text('营业成本 2,229 百万元', encoding='utf-8')
    v = CitationValidator(corpus_root=str(tmp_path))
    rec = v.check_reference(str(f), '营业成本 22.29 亿元', accept_unit_variants=True)
    assert rec['num_hit'] == 1 and '22.29' in rec['unit_hit']
    rec2 = v.check_reference(str(f), '营业成本 22.29 亿元')
    assert rec2['num_hit'] == 0 and '22.29' in rec2['unhit']


def test_year_pair_not_merged():
    # “2023 2024”年份并列不应被折叠合并成 8 位数字；普通 LaTeX 空格数字仍折叠
    v = CitationValidator()
    assert v.extract_numbers('2023 2024 年营业收入') == ['2023', '2024']
    assert v.extract_numbers('区间从 1 200 点到 2 5 0 0 点') == ['1200', '2500']


def test_table_row_not_merged():
    # 表格行：空格分隔的多个数字不应被合并（折叠仅限单数字 LaTeX 排版空格）
    v = CitationValidator()
    assert v.extract_numbers('0.72 2.73 3.71 5.08 5.08 139.37') == [
        '0.72', '2.73', '3.71', '5.08', '5.08', '139.37',
    ]
    assert v.extract_numbers('57.2 1019.9 2.66 2.91') == ['57.2', '1019.9', '2.66', '2.91']


def test_hits_context_extraction(tmp_path):
    # 命中数字应在原始文件文本中定位并附带上下文（供前端"查看原文"展示）
    f = tmp_path / "云南白药2024年报.md"
    f.write_text(
        "云南白药2024年实现营业收入400.33亿元，同比增长2.36%，归母净利润47.49亿元，同比增长16.02%。",
        encoding="utf-8",
    )
    v = CitationValidator(corpus_root=str(tmp_path))
    rec = v.check_reference(str(f), "归母净利润47.49亿元，同比增长16.02%")
    assert rec["num_hit"] == rec["nums"]
    assert rec["hits_context"]
    assert all("num" in h and "context" in h for h in rec["hits_context"])
    assert any(h["num"] == "47.49" for h in rec["hits_context"])
    ctx = [h for h in rec["hits_context"] if h["num"] == "47.49"][0]["context"]
    assert "47.49" in ctx and "云南白药" in ctx
    # 条数上限
    assert len(rec["hits_context"]) <= v.MAX_HIT_CONTEXTS


def test_hits_context_skips_unnormalized_number(tmp_path):
    # 仅靠排版空格折叠（8 7 . 5 -> 87.5）才命中的数字，原始文本中定位不到则无上下文（不报错）
    f = tmp_path / "康美指数.md"
    f.write_text("区间价格涨幅达$8 7 . 5 \\%$，指数从 1 200 点上扬", encoding="utf-8")
    v = CitationValidator(corpus_root=str(tmp_path))
    rec = v.check_reference(str(f), "涨幅 $8 7 . 5\\%$")
    assert rec["num_hit"] == rec["nums"] == 1
    assert isinstance(rec["hits_context"], list)
