from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    # Elasticsearch
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "bank_rules"

    # RAG settings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    chunk_size: int = 200
    chunk_overlap: int = 20
    top_k: int = 3

    # Paths
    documents_path: str = "./data/bank_rules/"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
