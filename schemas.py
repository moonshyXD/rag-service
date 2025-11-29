from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Запрос на поиск контекста"""
    query: str
    top_k: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Какой срок ответа на запрос информации?",
                "top_k": 3
            }
        }


class ContextResponse(BaseModel):
    """Ответ с найденным контекстом"""
    context: str
    sources: list[str]
    relevance_scores: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "context": "Правило 3.2: Срок ответа - 3 рабочих дня...",
                "sources": ["compliance_policy.pdf", "internal_procedures.pdf"],
                "relevance_scores": [0.92, 0.87, 0.81]
            }
        }


class IndexRequest(BaseModel):
    """Запрос на индексацию документов"""
    documents_path: str = "./data/bank_rules/"
    force_reindex: bool = False


class IndexResponse(BaseModel):
    """Результат индексации"""
    status: str
    documents_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Статус сервиса"""
    status: str
    elasticsearch_connected: bool
    documents_count: int
