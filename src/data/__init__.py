from .loader import load_markdown_documents, load_industry_documents
from .metadata import load_excel_metadata_by_title, get_best_metadata_for_title

try:
    from .splitter import (
        split_documents,
        recursive_split_text,
        extract_tables_and_text,
        parse_html_table,
        _parse_html_table_regex,
        parse_markdown_table,
    )
except ImportError:
    # langchain_text_splitters may not be installed in the current environment.
    pass
