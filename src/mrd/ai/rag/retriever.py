from mrd.ai.rag.embeddings import EmbeddingService
from mrd.ai.rag.store import VectorStore


class RAGRetriever:

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingService,
    ) -> None:

        self.store = store
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        embedding = self.embeddings.embed(
            query
        )

        return self.store.search(
            embedding=embedding,
            limit=limit,
        )

    def build_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        documents = self.retrieve(
            query=query,
            limit=limit,
        )

        if not documents:
            return "No relevant knowledge found."

        chunks = []

        for document in documents:

            chunks.append(
                f"""
SOURCE: {document["source"]}
TITLE: {document["title"]}
SIMILARITY: {document["similarity"]}

{document["content"]}
"""
            )

        return "\n---\n".join(chunks)