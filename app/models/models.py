from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document


class InitCollectionRequest(BaseModel):
    collection_name: str


class StoreDocsRequest(BaseModel):
    collection_name: str
    file_paths: List[str]


class QueryRequest(BaseModel):
    collection_name: str
    query: str
    filenames: Optional[List[str]] = None
    k: int = 3


class RAGQueryMetadata(BaseModel):
    selected_files: Optional[list[str]] = None


class RAGQueryRequest(BaseModel):
    user_query: str
    collection_name: str
    metadata: Optional[RAGQueryMetadata] = None


class RAGQueryResponse(BaseModel):
    answer: str
    context: Optional[List[Document]] = None


class KnowledgeBotMetadata(BaseModel):
    selected_files: Optional[list[str]] = None


class KnowledgeBotRequest(BaseModel):
    user_query: str
    thread_id: str
    metadata: Optional[KnowledgeBotMetadata] = None
