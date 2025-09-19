from typing import Optional
from langgraph.graph import MessagesState


class KnowledgeBotState(MessagesState):
    selected_files: Optional[list[str]] = None
