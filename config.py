"""Config File.

DOCS_DIRECTORY: (str) : Directory with your documentation.
DATABASE_DIRECTORY: (str) : Directory with your database.
DATABASE_CREATION (bool): if True, the vector database is created: fetch the documents, split them in chunks and embed them.
If False, just retrieve the vector database with the embeddings, ready for query.
EMBEDDING_MODEL (str): give the Ollama model for embedding.
LLM_MODEL (str): give the Ollama model for LLM task.
RAG_LLM (bool): Choose between RG+LLM and just information retrieval.
"""
DOCS_DIRECTORY: str = "./docs"
DATABASE_DIRECTORY: str = "./db"
DATABASE_CREATION: bool = False
EMBEDDING_MODEL: str = "nomic-embed-text"
LLM_MODEL: str = "qwen2:7b"
RAG_LLM: bool = False
