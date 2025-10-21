"""
使用嵌入式向量数据库，将文件保存到向量数据库以供检索使用
"""

import os
import logging
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def load(src_dir, store_dir, glob="**/*.md", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # 1. 加载 Markdown 文件
    loader = DirectoryLoader(
        path=src_dir,
        glob=glob,
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    documents = loader.load()
    logger.info("## file has loaded")

    # 2. 分块（结合 Markdown 标题和字符分割）
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in documents:
        md_chunks = markdown_splitter.split_text(doc.page_content)
        for chunk in md_chunks:
            chunk.metadata.update(doc.metadata)
            chunks.extend(text_splitter.split_documents([chunk]))
    logger.info("## file has chunked")

    # 3. 嵌入和向量存储
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=store_dir
    )
    logger.info("## file has saved to vetor store %s", store_dir)
    return vectorstore
