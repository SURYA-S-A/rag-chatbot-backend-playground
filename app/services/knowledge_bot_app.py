from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.prompts import MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from app.models.models import (
    KnowledgeBotRequest,
)
from app.models.knowledge_bot_pipeline import KnowledgeBotState
from app.services.llm_manager import LLMManager
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, StateGraph
from IPython.display import Image, display
from app.services.knowledge_bot_tools import KnowledgeTools


memory_saver = MemorySaver()


class KnowledgeBotApp:
    def __init__(self, llm_manager: LLMManager) -> None:
        self.llm_manager = llm_manager
        self.knowledge_tools: KnowledgeTools = KnowledgeTools()
        self.compiled_graph: CompiledStateGraph = self._build_rag_graph()

    def _agent(self, state: KnowledgeBotState):
        llm = self.llm_manager.get_model()
        agent_prompt = ChatPromptTemplate(
            [
                SystemMessage(
                    content="""
                    You are a Knowledge Bot that helps users find information from their uploaded documents.
                    IMPORTANT: Always try to search the document collection first using the rag_retrival tool when users ask questions. Only ask them to upload documents if the RAG search returns no results or if there are clearly no documents in the system.
                    Don't ask users to upload documents unless you've first attempted to search the existing document collection.
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        agent_runnable = agent_prompt | llm.bind_tools(
            tools=[self.knowledge_tools.calculator, self.knowledge_tools.rag_retrival],
        )
        response = agent_runnable.invoke(input=state)
        state["messages"] = add_messages(left=state["messages"], right=response)
        return state

    def _build_rag_graph(self) -> CompiledStateGraph:
        graph = StateGraph(KnowledgeBotState)

        graph.add_node("agent", self._agent)
        graph.add_node(
            "tools",
            ToolNode(
                [self.knowledge_tools.calculator, self.knowledge_tools.rag_retrival]
            ),
        )

        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph.add_edge("tools", "agent")

        compiled_graph = graph.compile(checkpointer=memory_saver)
        return compiled_graph

    def display_graph(self) -> None:
        display(
            Image(
                self.compiled_graph.get_graph(xray=True).draw_mermaid_png(max_retries=5)
            )
        )

    def run_agent(self, knowledge_bot_request: KnowledgeBotRequest) -> None:
        config = RunnableConfig(
            configurable={"thread_id": knowledge_bot_request.thread_id}
        )
        result = self.compiled_graph.invoke(
            {
                "messages": HumanMessage(content=knowledge_bot_request.user_query),
                "selected_files": knowledge_bot_request.metadata.selected_files,
            },
            config=config,
        )
        print(result["messages"][-1].content)
        return result["messages"][-1].content
