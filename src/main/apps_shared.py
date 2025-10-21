'''共享变量'''
from pathlib import Path

STORE_PATH=Path(__file__).parent.parent.parent / "tmp" / "vector_store"
EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"
