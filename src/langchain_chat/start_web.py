#! /usr/bin/env python

'''启动web应用, 其中后每次发送消息都会执行src/start_web.py整个文件， 这里需要设计 APP为单例避免重复初始化
bash: streamlit run src/start_web.py config .streamlit/config.toml
'''
from langchain_chat.web_app import APP

APP.render_ui()
