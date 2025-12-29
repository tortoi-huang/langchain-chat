import logging
from typing import TypeVar
import os

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import langchain_chat.app_config.app_config as app_config

logger = logging.getLogger(__name__)
T = TypeVar("T")

class MyChat:

    """基于向量数据库和通义千问模型的RAG问答系统"""
    def __init__(
            self,
            api_key,
            vetor_store,
            model="qwen-plus",
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2"):
        """gpt-3.5-turbo, qwen-plus, qwen-max"""
        # 设置 DashScope API 密钥
        # os.environ["DASHSCOPE_API_KEY"] = "sk-5a0c9680ce1948e2b4c4a533325b6c36"

        # 1. 加载现有向量存储
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        logger.info("## embeddings_model: %s", embeddings_model)
        # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        self.vectorstore = Chroma(persist_directory=vetor_store, embedding_function=embeddings)

        # 2. 设置 RAG 链
        self.llm = ChatTongyi(model=model, api_key=api_key)  # 使用 Qwen-Max 模型
        logger.info("## model: %s", model)

        template = """基于以下上下文回答问题：
{context}

问题：{question}"""
        prompt = PromptTemplate.from_template(template)
        # 创建检索器，
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", # “相似性搜索”（cosine similarity，余弦相似度
            search_kwargs={"k": 5},   # 表示返回 top-5 个最相似的文档片段（chunks）
            )
        
        def chan_logger(d: T) -> T:
            logger.info("## RAG链参数: %s", d)
            return d

        # 下面竖线"|"是langchain定义的管道操作符，与unix一致，前面函数的输出放到后面函数的输入
        # 当调用 rag_chain.invoke("my quest") 时，retriever 查询 "my quest" 得到相关文档列表 docs, 然后调用lambda处理，封装成一个dict，并传递给prompt函数
        self.rag_chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
                "question": RunnablePassthrough()
            }
            | prompt
            | chan_logger
            | self.llm
            | StrOutputParser()
        )

    def query(self, quest: str) -> str:
        """提问"""
        logger.info("## 问题：%s", quest)
        if not quest.strip():
            return "请输入问题："
        return self.rag_chain.invoke(quest)

    def call_api(self, quest: str) -> str:
        """提问"""
        logger.info("## 问题：%s", quest)
        if not quest.strip():
            return "请输入问题："
        res = self.llm.stream([HumanMessage(content=quest)], streaming=True)
        answer = ''
        for r in res:
            # print(r.content, end='')
            answer += str(r.content)
        return answer
    def count_store_item(self):
        return self.vectorstore._collection.count()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("错误：未找到 DASHSCOPE_API_KEY，请设置环境变量或在 .env 文件中配置")
        exit(1)
    
    if not app_config.STORE_PATH.exists():
        print(f"错误：未找到向量存储路径 {app_config.STORE_PATH}，请先运行 vetor_store.py 进行数据加载")
        exit(1)
    print("## vetor_store path: ", app_config.STORE_PATH)
    chat = MyChat(os.environ["DASHSCOPE_API_KEY"], str(app_config.STORE_PATH))
    while True:
        cli_question = input("请输入您的问题（输入 '退出' 或 'quit' 退出）：")
        if cli_question.lower() in ["退出", "quit"]:
            break
        response = chat.query(cli_question)
        print("回答：", response, "\n")
