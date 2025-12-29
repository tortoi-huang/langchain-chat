import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import langchain_chat.app_config.app_config as app_config
from langchain_chat.main.chat_agent import MyChat

app = FastAPI(title="RAG Chatbot with Qwen-Plus", version="1.0")

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
    try:
        response = chatbot.query(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen API 调用失败: {str(e)}") from e

    return ChatResponse(session_id=session_id, reply=response)


@app.get("/health")
def health():
    count = chatbot.count_store_item()
    return {"status": "ok", "vector_db_doc_count": count, "model": "qwen-plus"}
