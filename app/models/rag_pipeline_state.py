from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class RAGPipelineState(TypedDict):
    question: str
    answer: str
    context: List[Document]
    collection_name: str
    selected_files: Optional[list[str]] = None
