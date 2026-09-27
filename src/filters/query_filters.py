from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class QueryFilters:
    stock_name: Optional[str] = None
    stock_code: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rating: Optional[str] = None
    org_name: Optional[str] = None
    researcher: Optional[str] = None
    industry: Optional[str] = None
    title: Optional[str] = None
    doc_type: Optional[str] = None

    def compute_match_score(self, payload: Dict[str, Any]) -> float:
        """
        计算 payload 与过滤条件的匹配得分（0~1）。
        每项匹配权重可自定义，目前简单按匹配项数量归一化。
        """
        if not self.has_any_filter():
            return 1.0  # 无任何过滤条件，默认满分

        total_checks = 0
        matched = 0

        # 股票名称或代码：只要有一个匹配即可（权重可调高）
        if self.stock_name or self.stock_code:
            total_checks += 2  # 权重较高
            if self.stock_name and self.stock_name == payload.get("stockName", ""):
                matched += 2
            elif self.stock_code and self.stock_code == payload.get("stockCode", ""):
                matched += 2

        # 评级
        if self.rating:
            total_checks += 1
            if self.rating == payload.get("emRatingName", ""):
                matched += 1

        # 券商
        if self.org_name:
            total_checks += 1
            if self.org_name == payload.get("orgName", ""):
                matched += 1

        # 研究员（支持部分匹配）
        if self.researcher:
            total_checks += 1
            if self.researcher in payload.get("researcher", ""):
                matched += 1

        # 行业
        if self.industry:
            total_checks += 1
            if self.industry == payload.get("indvInduName", ""):
                matched += 1

        # 标题
        if self.title:
            total_checks += 1
            if self.title in payload.get("title", ""):
                matched += 1

        # 文档类型
        if self.doc_type:
            total_checks += 1
            if self.doc_type == payload.get("doc_type", ""):
                matched += 1

        # 日期范围（暂不纳入，可自行扩展）
        # ...

        return matched / total_checks if total_checks > 0 else 1.0

    def has_any_filter(self) -> bool:
        """判断是否有任何过滤条件被设置"""
        return any([
            self.stock_name, self.stock_code, self.start_date, self.end_date,
            self.rating, self.org_name, self.researcher, self.industry,
            self.title, self.doc_type
        ])
