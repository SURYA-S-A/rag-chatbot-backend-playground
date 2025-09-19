from typing import Any, Dict
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image, display
from app.models.rag_pipeline_state import RAGPipelineState
from app.models.models import (
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.llm_manager import LLMManager
from app.services.vector_store_manager import VectorStoreManager


class RAGPipeline:
    def __init__(
        self, llm_manager: LLMManager, vector_store_manager: VectorStoreManager
    ) -> None:
        self.llm_manager = llm_manager
        self.vector_store_manager = vector_store_manager
        self.compiled_graph: CompiledStateGraph = self._build_rag_graph()

    def _retrieve_documents(self, state: RAGPipelineState) -> Dict[str, Any]:
        retrieved_docs = self.vector_store_manager.query(
            collection_name=state["collection_name"],
            query=state["question"],
            selected_files=state["selected_files"],
        )
        return {"context": retrieved_docs}

    def _generate_answer(self, state: RAGPipelineState):
        llm = self.llm_manager.get_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant. 
                    Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    {context}
                    """,
                ),
                ("user", "\nQuestion: {question}"),
            ]
        )

        context_texts = [doc.page_content for doc in state["context"]]
        context_combined = "\n\n".join(context_texts)

        chat_input = prompt.format_prompt(
            context=context_combined,
            question=state["question"],
        ).to_messages()

        answer = llm.invoke(chat_input).content
        return {"answer": answer}

    def _build_rag_graph(self) -> CompiledStateGraph:
        graph = StateGraph(RAGPipelineState)

        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("generate_answer", self._generate_answer)

        graph.add_edge(START, "retrieve_documents")
        graph.add_edge("retrieve_documents", "generate_answer")
        graph.add_edge("generate_answer", END)

        compiled_graph = graph.compile()
        return compiled_graph

    def display_graph(self) -> None:
        display(Image(self.compiled_graph.get_graph().draw_mermaid_png()))

    def run_pipeline(self, rag_query_request: RAGQueryRequest) -> RAGQueryResponse:
        result = self.compiled_graph.invoke(
            {
                "question": rag_query_request.user_query,
                "collection_name": rag_query_request.collection_name,
                "selected_files": rag_query_request.metadata.selected_files,
            }
        )
        print(f"Rag Bot answer - {result["answer"]}")
        return RAGQueryResponse(answer=result["answer"], context=result.get("context"))
