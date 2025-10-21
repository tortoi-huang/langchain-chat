# 项目介绍
一个简单的RAG 项目， 包括两个进程:
1. load2vetor.py: 将资料加载到向量数据库, 包括拆分chunk和计算向量
2. start_web.py: 启动web聊天服务器，聊天时通过输入消息去向量数据库查找近似的资料重新组织消息发送给大模型提问

add:
- cmmi_doc2txt.ipynb: 将word, excel, pdf转换为文本
# 启动web
```bash
streamlit run src/web_app.py config .streamlit/config.toml
```

# streamlit
streamlit 是一个简单的单用户 ai聊天web框架，

# python相关
更新所有模块
```bash
uv sync
```

添加依赖模块
```bash
# 多个包名用空格分开
uv add langchain_chroma langchain_huggingface langchain_community
uv add unstructured unstructured[md] langchain chromadb sentence-transformers groq
uv add langchain_groq


# 添加开发时使用的包，多个包名用空格分开
uv add --dev pytest mypy
```


