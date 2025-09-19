from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.config.settings import settings


class VectorStoreManager:
    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self.client = QdrantClient(
            api_key=settings.QDRANT_API_KEY,
            url=settings.QDRANT_URL,
        )
        self.embeddings = embeddings

    def _get_vector_size(self) -> int:
        """Get embedding vector size dynamically."""
        return len(self.embeddings.embed_query("dimension check"))

    def init_collection(self, collection_name: str) -> None:
        """Create collection and indexes if not exists."""
        if not self.client.collection_exists(collection_name):
            vector_size = self._get_vector_size()
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection '{collection_name}'")

            # Create payload index
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.filename",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        else:
            print(f"Using existing collection '{collection_name}'")

    def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Return LangChain Qdrant vector store wrapper."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, collection_name: str, docs: list[Document]) -> None:
        """Insert documents into vector store."""
        vs = self.get_vector_store(collection_name)
        vs.add_documents(docs)
        print(f"Uploaded {len(docs)} chunks to '{collection_name}'")

    def query(
        self,
        collection_name: str,
        query: str,
        selected_files: list[str] | None = None,
        k: int = 3,
    ) -> List[Document]:
        """Search collection with optional filename filter."""
        vs = self.get_vector_store(collection_name)

        filter_ = None
        if selected_files:
            filter_ = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchAny(any=selected_files),
                    )
                ]
            )

        results = vs.similarity_search(query, k=k, filter=filter_)
        return results
