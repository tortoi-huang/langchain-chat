#! /usr/bin/env python

'''加载知识库到向量存储'''
import logging
from pathlib import Path
from langchain_chat.main.vetor_store import load
import langchain_chat.app_config.app_config as app_config

logging.basicConfig(
    level=logging.DEBUG, format="## %(asctime)s %(name)s %(levelname)s %(message)s"
)

load(
    src_dir=app_config.DOC_PATH,
    store_dir=app_config.STORE_PATH.resolve(),
    glob="**/*.md",
    model_name=app_config.EMBEDDINGS_MODEL,
)
