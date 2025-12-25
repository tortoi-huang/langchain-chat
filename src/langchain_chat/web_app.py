import logging
import os
import sys

import streamlit as st

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@st.cache_resource
def get_chat_client():
    """"""
    import langchain_chat.app_config.app_config as app_config
    from langchain_chat.main.query_rag import MyChat

    if "DASHSCOPE_API_KEY" not in os.environ:
        logger.error("未找到 DASHSCOPE_API_KEY，请设置环境变量或在 .env 文件中配置")
        raise EnvironmentError("DASHSCOPE_API_KEY not found in environment variables")

    store_path = app_config.STORE_PATH
    if not store_path.exists():
        logger.error(
            "未找到向量存储路径 %s，请先运行 vetor_store.py 进行数据加载",
            store_path,
        )
        raise FileNotFoundError(f"Vector store path {store_path} not found")
    logger.info("## vetor_store path: %s", store_path)
    return MyChat(
        os.environ["DASHSCOPE_API_KEY"],
        store_path.resolve(),
        embeddings_model=app_config.EMBEDDINGS_MODEL,
    )


chatbot = get_chat_client()

if "chat_history" not in st.session_state:
    """渲染 Streamlit 界面"""
    system_prompt = "假如你是软件工程标准化过程CMMI专家"
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("📚 RAG 问答机器人")
    # st.markdown("使用通义千问和向量数据库进行智能问答")
    st.caption("使用您的 Markdown 文件提问，Qwen-Max 将基于检索内容回答。")
    st.session_state.chat_history = [("system", system_prompt)]


# 显示历史消息
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input(placeholder="请输入您的问题...")
if not user_input:
    sys.exit(0)
# 显示用户消息
st.chat_message("user").markdown(user_input)
st.session_state.chat_history.append(("user", user_input))

# 获取机器人回复
with st.chat_message("assistant"):
    with st.spinner("正在生成回答..."):
        response = chatbot.query(user_input)
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))

# bash: streamlit run src/web_app.py config .streamlit/config.toml
# if __name__ == "__main__":
#     APP.render_ui()
