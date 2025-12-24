from pathlib import Path

config = {
    # "api_key": "",
    "embed_module": "sentence-transformers/all-MiniLM-L6-v2",
    # base on app.py
    "vetor_store": "../tmp/vector_store",
}


BASE_PATH = Path(__file__).parent.parent.parent.parent
DOC_PATH = BASE_PATH / "data" / "cmmi"
STORE_PATH = BASE_PATH / "tmp" / "vector_store"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
