from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack import Document
from pathlib import Path
from settings import Settings
import logging

logger = logging.getLogger(__name__)


def convert_files_to_docs(dir_path: str) -> list[Document]:
    """
    Простейший аналог convert_files_to_docs.
    Читает все .txt файлы из папки и превращает их в Haystack Document.
    """
    docs: list[Document] = []
    base = Path(dir_path)

    if not base.exists():
        return docs

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        docs.append(
            Document(
                content=text,
                meta={"file_path": str(path)},
            )
        )

    return docs


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings

        # Document store (Elasticsearch)
        self.document_store = ElasticsearchDocumentStore(
            hosts=f"http://{settings.es_host}:{settings.es_port}",
            index=settings.es_index,
        )

        # Embedder для документов
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model
        )
        self.doc_embedder.warm_up()

        # Embedder для запросов
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=settings.embedding_model
        )
        self.text_embedder.warm_up()

        # Retriever поверх ElasticsearchDocumentStore
        self.retriever = ElasticsearchEmbeddingRetriever(
            document_store=self.document_store
        )

        # Preprocessors
        self.cleaner = DocumentCleaner()
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        )

    def index_documents(self, docs_path: str, force_reindex: bool = False) -> int:
        """Индексация документов"""
        try:
            if force_reindex:
                logger.info("Очистка существующего индекса...")
                self.document_store.delete_documents()

            logger.info(f"Загрузка документов из {docs_path}...")
            docs = convert_files_to_docs(dir_path=docs_path)

            if not docs:
                raise ValueError(f"Не найдено документов в {docs_path}")

            logger.info("Обработка документов...")
            docs = self.cleaner.run(documents=docs)["documents"]
            docs = self.splitter.run(documents=docs)["documents"]

            logger.info("Создание embeddings...")
            docs_with_embeddings = self.doc_embedder.run(documents=docs)["documents"]

            logger.info("Сохранение в Elasticsearch...")
            self.document_store.write_documents(docs_with_embeddings)

            logger.info(
                f"Успешно проиндексировано {len(docs_with_embeddings)} фрагментов"
            )
            return len(docs_with_embeddings)

        except Exception as e:
            logger.error(f"Ошибка индексации: {str(e)}")
            raise

    def get_context(self, query: str, top_k: int = 3) -> dict:
        """Получение релевантного контекста"""
        try:
            query_embedding = self.text_embedder.run(text=query)["embedding"]

            results = self.retriever.run(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            documents = results["documents"]

            if not documents:
                return {
                    "context": "Релевантных правил не найдено.",
                    "sources": [],
                    "relevance_scores": [],
                }

            context_parts: list[str] = []
            sources: list[str] = []
            scores: list[float] = []

            for doc in documents:
                source = doc.meta.get("file_path", "unknown")
                score = doc.score if hasattr(doc, "score") else 0.0

                context_parts.append(f"{doc.content}\n(Источник: {source})")
                sources.append(source)
                scores.append(score)

            return {
                "context": "\n\n---\n\n".join(context_parts),
                "sources": sources,
                "relevance_scores": scores,
            }

        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            raise

    def get_documents_count(self) -> int:
        """Количество документов в индексе"""
        try:
            return self.document_store.count_documents()
        except Exception:
            return 0

    def check_connection(self) -> bool:
        """Проверка подключения к Elasticsearch"""
        try:
            self.document_store.count_documents()
            return True
        except Exception:
            return False
