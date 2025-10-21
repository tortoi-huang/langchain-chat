#! /usr/bin/env python

'''加载知识库到向量存储'''
import logging
from pathlib import Path
from main.vetor_store import load
import main.apps_shared as shared

logging.basicConfig(
    level=logging.DEBUG, format="## %(asctime)s %(name)s %(levelname)s %(message)s"
)

load(
    Path(__file__).parent.parent / "data" / "cmmi",
    shared.STORE_PATH.resolve(),
    glob="**/*.md",
    model_name=shared.EMBEDDINGS_MODEL,
)
