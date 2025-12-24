# 项目介绍
一个简单的RAG 项目， 包括两个进程:
1. load2vetor.py: 将资料加载到向量数据库, 包括拆分chunk和计算向量并保持到向量数据库中
2. start_web.py: 启动web聊天服务器，聊天时通过输入消息去向量数据库查找近似的资料重新组织消息发送给大模型提问

add:
- cmmi_doc2txt.ipynb: 将word, excel, pdf转换为文本
# 启动web

## streamlit ui
配置 .streamlit\config.toml
```bash
# streamlit 每发送一条消息都会从头到尾执行一次src/web_app.py, 对于初始化代码需要使用单例模式
export DASHSCOPE_API_KEY="api key"
# $env:DASHSCOPE_API_KEY="api key"
# 启动web服务
streamlit run src/langchain_chat/web_app.py config .streamlit/config.toml
```
打开浏览器: http://localhost:8080/

## web api 服务

```bash
export DASHSCOPE_API_KEY="api key"
uvicorn langchain_chat.pydantic_web:app --host 0.0.0.0 --port 8000 --reload

curl -X POST -H "Content-Type: application/json" -d '{"message": "你好,请问EPG的职责是什么？"}' http://localhost:8000/chat
```

# streamlit
streamlit 是一个简单的单用户 ai聊天web框架，

# Chroma 向量数据库
Chroma 是一个向量数据库，可以存储向量数据，并提供向量相似度查询接口, 这里使用langchain_chroma 来实现。

Chroma 向量数据库是是基于 sqlite 实现的嵌入式数据库。

# python相关
更新下载 pyproject.toml 中所有模块
```bash
uv sync
```

添加依赖模块
```bash
# 使用uv add 将包依赖添加到 pyproject.toml 中，多个包名用空格分开
uv add langchain_chroma langchain_huggingface langchain_community
uv add unstructured unstructured[md] langchain chromadb sentence-transformers groq
uv add langchain_groq


# 添加开发时使用的包，多个包名用空格分开
uv add --dev pytest mypy

# uv首次运行uv sync时会固定依赖版本到uv.lock上, 更新依赖最新版本并更新uv.lock
uv sync --upgrade

# 运行脚本
uv run .\src\langchain_chat\load2vetor.py
```


# Hugging Face 镜像
Hugging Face可能被墙，无法下载模型，需要配置镜像地址
```sh
export HF_ENDPOINT=https://hf-mirror.com
# $env:HF_ENDPOINT="https://hf-mirror.com"
```

