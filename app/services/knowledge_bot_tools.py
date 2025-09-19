from typing import Annotated
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from app.models.models import RAGQueryMetadata, RAGQueryRequest
from app.services.rag_pipeline import RAGPipeline
from langchain_core.runnables import RunnableConfig


class KnowledgeTools:
    rag_pipeline: RAGPipeline = None

    @classmethod
    def set_rag_pipeline(cls, rag_pipeline):
        cls.rag_pipeline = rag_pipeline

    @tool()
    def calculator(
        a: int,
        b: int,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,
    ) -> Command:
        """
        Use this tool to add two numbers.
        """
        result = a + b
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    @tool()
    def rag_retrival(
        user_query: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,
        state: Annotated[dict, InjectedState] = None,
        special_config_param: RunnableConfig = None,
    ) -> Command:
        """
        Use this tool to search and give details about the documents uploaded by the user.
        """
        thread_id = special_config_param["configurable"]["thread_id"]
        rag_bot_request = RAGQueryRequest(
            user_query=user_query,
            collection_name=thread_id,
            metadata=RAGQueryMetadata(selected_files=state["selected_files"]),
        )
        result = KnowledgeTools.rag_pipeline.run_pipeline(rag_bot_request)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )
