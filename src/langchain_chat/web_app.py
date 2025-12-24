import logging
import os

import streamlit as st

import langchain_chat.app_config.app_config as app_config
from langchain_chat.main.query_rag import MyChat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyStreamlitApp:
    """Streamlit 应用程序类"""

    def __init__(self):
        """初始化聊天机器人"""
        if "DASHSCOPE_API_KEY" not in os.environ:
            logger.error("未找到 DASHSCOPE_API_KEY，请设置环境变量或在 .env 文件中配置")
            raise EnvironmentError(
                "DASHSCOPE_API_KEY not found in environment variables"
            )

        store_path = app_config.STORE_PATH
        if not store_path.exists():
            logger.error(
                "未找到向量存储路径 %s，请先运行 vetor_store.py 进行数据加载",
                store_path,
            )
            raise FileNotFoundError(f"Vector store path {store_path} not found")
        logger.info("## vetor_store path: %s", store_path)
        self.chatbot = MyChat(
            os.environ["DASHSCOPE_API_KEY"],
            store_path.resolve(),
            embeddings_model=app_config.EMBEDDINGS_MODEL,
        )
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        logger.info("## app started")

    def render_ui(self):
        """渲染 Streamlit 界面"""
        st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
        st.title("📚 RAG 问答机器人")
        st.markdown("使用通义千问和向量数据库进行智能问答")
        st.caption("使用您的 Markdown 文件提问，Qwen-Max 将基于检索内容回答。")

        # 下面代码输入框在靠下左右居中, 聊天记录在上
        # 显示历史消息
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        # 聊天输入框（固定在页面底部）
        user_input = st.chat_input("请输入您的问题...")

        if user_input:
            self.handle_query(user_input)

    def handle_query(self, user_input):
        """处理用户查询"""
        # 显示用户消息
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        # 获取机器人回复
        with st.chat_message("assistant"):
            with st.spinner("正在生成回答..."):
                response = self.chatbot.query(user_input)
                st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))


APP = MyStreamlitApp()
# bash: streamlit run src/web_app.py config .streamlit/config.toml
if __name__ == "__main__":
    APP.render_ui()
