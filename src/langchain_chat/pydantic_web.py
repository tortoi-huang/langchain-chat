import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel

import langchain_chat.app_config.app_config as app_config
from langchain_chat.main.query_rag import MyChat

app = FastAPI(title="RAG Chatbot with Qwen-Plus", version="1.0")

# 会话存储（生产环境建议用 Redis）
sessions = {}

chatbot = MyChat(
    os.environ["DASHSCOPE_API_KEY"],
    app_config.STORE_PATH.resolve(),
    embeddings_model=app_config.EMBEDDINGS_MODEL,
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # 获取或创建会话记忆
    if session_id not in sessions:
        memory = ConversationBufferMemory(memory_key="history", return_messages=False)
        sessions[session_id] = memory
    else:
        memory = sessions[session_id]

    # 获取历史对话字符串
    # history = memory.load_memory_variables({})["history"]

    try:
        response = chatbot.query(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen API 调用失败: {str(e)}")

    # 保存当前对话到记忆
    memory.save_context({"input": request.message}, {"output": response})

    return ChatResponse(session_id=session_id, reply=response)


@app.get("/health")
def health():
    count = chatbot.vectorstore._collection.count()
    return {"status": "ok", "vector_db_doc_count": count, "model": "qwen-plus"}
