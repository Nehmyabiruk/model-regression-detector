from pydantic import BaseModel


class KnowledgeDocument(BaseModel):
    document_id: str
    title: str
    content: str
    source: str
    metadata: dict = {}