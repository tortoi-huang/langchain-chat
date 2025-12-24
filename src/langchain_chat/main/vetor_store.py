"""
使用嵌入式向量数据库，将文件保存到向量数据库以供检索使用
"""

import logging

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


def load(
    src_dir,
    store_dir,
    glob="**/*.md",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """加载 Markdown 文件，分块处理并存储为向量数据库。
    Args:
        src_dir (str): 包含 Markdown 文件的源目录路径。
        store_dir (str): 向量数据库的存储目录路径。
        glob (str, optional): 文件匹配模式，默认为 "**/*.md"。
        model_name (str, optional): 使用的嵌入模型名称，默认为 "sentence-transformers/all-MiniLM-L6-v2"。
    Returns: Chroma: 加载并存储后的向量数据库对象。
    Raises: FileNotFoundError: 如果源目录不存在或无法访问。 ValueError: 如果分块或嵌入过程中出现错误。
    """
    # 1. 加载 Markdown 文件
    logger.info("## loading files %s to vetor store %s", src_dir, store_dir)
    loader = DirectoryLoader(
        path=src_dir,
        glob=glob,
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    documents = loader.load()
    logger.info("## file %s has loaded", src_dir)

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

    # 3. 嵌入和向量存储，这里需要到huggingFace下载模型，确保网络通常
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=store_dir,
        # collection_name="langchain_chat", # collection_name默认为langchain
    )
    logger.info("## file has saved to vetor store %s", store_dir)
    return vectorstore
