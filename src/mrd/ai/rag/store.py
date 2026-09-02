from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text


@dataclass
class VectorDocument:
    document_id: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any]
    embedding: list[float]


class VectorStore:
    """PostgreSQL + pgvector document store."""

    def __init__(
        self,
        database_url: str,
    ) -> None:

        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
        )

        self._initialize()

    def _initialize(self) -> None:

        with self.engine.begin() as connection:

            connection.execute(
                text(
                    """
                    CREATE EXTENSION IF NOT EXISTS vector;
                    """
                )
            )

            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS
                    mrd_knowledge_documents (
                        document_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        embedding vector(384)
                    );
                    """
                )
            )

    def upsert(
        self,
        document: VectorDocument,
    ) -> None:

        with self.engine.begin() as connection:

            connection.execute(
                text(
                    """
                    INSERT INTO mrd_knowledge_documents
                    (
                        document_id,
                        title,
                        content,
                        source,
                        metadata,
                        embedding
                    )
                    VALUES (
                        :document_id,
                        :title,
                        :content,
                        :source,
                        CAST(:metadata AS jsonb),
                        CAST(:embedding AS vector)
                    )
                    ON CONFLICT (document_id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """
                ),
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "content": document.content,
                    "source": document.source,
                    "metadata": __import__(
                        "json"
                    ).dumps(document.metadata),
                    "embedding": str(
                        document.embedding
                    ),
                },
            )

    def search(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[dict]:

        with self.engine.begin() as connection:

            rows = connection.execute(
                text(
                    """
                    SELECT
                        document_id,
                        title,
                        content,
                        source,
                        metadata,
                        1 - (
                            embedding <=> CAST(
                                :embedding AS vector
                            )
                        ) AS similarity
                    FROM mrd_knowledge_documents
                    ORDER BY embedding <=> CAST(
                        :embedding AS vector
                    )
                    LIMIT :limit
                    """
                ),
                {
                    "embedding": str(embedding),
                    "limit": limit,
                },
            )

            return [
                dict(row._mapping)
                for row in rows
            ]