"""
langchain_rag_chain.py
使用 LangChain LCEL 构建 RAG 链路

通过 | 管道符依次串联：检索器 → 文档格式化 → Prompt → LLM → 输出解析器
"""

import logging
from typing import List

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from config.rag_config import get_config
from prompts.rag import build_rag_chat_prompt, format_history

logger = logging.getLogger(__name__)


def format_docs(docs):
    print(f"[DEBUG] 检索到 {len(docs)} 个文档")
    if not docs:
        print("[DEBUG] 文档列表为空")
        return "未找到相关内容"
    
    contents = []
    for i, doc in enumerate(docs):
        if hasattr(doc, 'page_content') and doc.page_content:
            contents.append(doc.page_content)
        else:
            print(f"[DEBUG] 文档 {i} 的 page_content 为空或不存在")
    
    if not contents:
        print("[DEBUG] 所有文档的 page_content 都为空")
        return "未找到相关内容"
    
    print(f"[DEBUG] 有效文档数: {len(contents)}，第一个文档前100字符: {contents[0][:100]}")
    return "\n\n---\n\n".join(contents)


def create_rag_chain(retriever):
    """创建 LCEL RAG 链。链路: retriever | format_docs | prompt | model | StrOutputParser"""
    cfg = get_config()
    model = cfg.get_chat_model()

    prompt = build_rag_chat_prompt()

    # ???retriever ????? question ?????? embed_query ??????? dict
    chain = (
        RunnablePassthrough.assign(
            context=lambda inputs: format_docs(retriever.invoke(inputs["question"])),
            history=lambda inputs: format_history(inputs.get("history")),
        )
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


class LangChainRAGChain:
    """封装 LCEL RAG 链的类，提供 invoke() 和 stream() 两种调用方式。"""

    def __init__(self, retriever):
        self.chain = create_rag_chain(retriever)

    def invoke(self, question: str) -> str:
        try:
            result = self.chain.invoke({"question": question})
            print(f"[DEBUG] LCEL 链返回结果长度: {len(result)}")
            print(f"[DEBUG] 结果内容: {repr(result)}")
            return result
        except Exception as e:
            print(f"[ERROR] LCEL 链执行异常: {e}")
            import traceback
            traceback.print_exc()
            return f"LCEL 链执行异常: {e}"

    def stream(self, question):
        return self.chain.stream(question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag_chain = LangChainRAGChain()
    answer = rag_chain.invoke("请简述一下通化金马最近一年的业绩情况")
    print("=== 回答 ===")
    print(answer)
