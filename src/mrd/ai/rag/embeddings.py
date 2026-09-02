import os

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate embeddings for RAG documents."""

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:

        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.model = SentenceTransformer(model_name)

    def embed(
        self,
        text: str,
    ) -> list[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()