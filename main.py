from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from settings import Settings, get_settings
from schemas import (
    QueryRequest, ContextResponse,
    IndexRequest, IndexResponse,
    HealthResponse,
)
from rag import RAGService
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Хендлер жизненного цикла: автоиндексация при старте."""
    settings: Settings = get_settings()
    rag = RAGService(settings)
    try:
        count = rag.index_documents(
            docs_path=settings.documents_path, 
            force_reindex=False,
        )
        logger.info(f"Автоиндексация при старте: {count} фрагментов")
    except Exception as e:
        logger.error(f"Автоиндексация не удалась: {e}")
    # даём приложению стартовать дальше
    yield
    # сюда можно добавить логику shutdown при необходимости


app = FastAPI(
    title="RAG Service",
    description="Сервис для поиска контекста из базы знаний банка",
    version="1.0.0",
    lifespan=lifespan,  # ВАЖНО: передаём lifespan вместо @on_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    """Singleton RAG сервиса"""
    return RAGService(settings)


@app.get("/", response_model=dict)
def root():
    """Корневой эндпоинт"""
    return {
        "service": "RAG Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
def health_check(
    rag: RAGService = Depends(get_rag_service)
):
    """Проверка здоровья сервиса"""
    es_connected = rag.check_connection()
    docs_count = rag.get_documents_count() if es_connected else 0

    return HealthResponse(
        status="healthy" if es_connected else "unhealthy",
        elasticsearch_connected=es_connected,
        documents_count=docs_count
    )


@app.post("/index", response_model=IndexResponse)
def index_documents(
    request: IndexRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """
    Индексация документов из указанной папки.
    Запускается при добавлении новых правил банка.
    """
    try:
        count = rag.index_documents(
            request.documents_path, 
            request.force_reindex
        )

        return IndexResponse(
            status="success",
            documents_indexed=count,
            message=f"Успешно проиндексировано {count} фрагментов документов"
        )

    except Exception as e:
        logger.error(f"Ошибка индексации: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка индексации документов: {str(e)}"
        )


@app.post("/query", response_model=ContextResponse)
def query_context(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """
    Поиск релевантного контекста по запросу.
    Основной эндпоинт для интеграции с другими сервисами.
    """
    try:
        result = rag.get_context(request.query, request.top_k)

        return ContextResponse(
            context=result["context"],
            sources=result["sources"],
            relevance_scores=result["relevance_scores"]
        )

    except Exception as e:
        logger.error(f"Ошибка поиска: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка поиска контекста: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
