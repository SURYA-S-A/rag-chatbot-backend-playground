from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from app.config.settings import settings


class LLMManager:
    """
    Handles initialization of LLMs with flexibility to switch models or providers.
    """

    def get_model(
        self, model_name: Optional[str] = None, model_provider: Optional[str] = None
    ) -> BaseChatModel:
        """
        Initialize and return the LLM instance.
        """
        return init_chat_model(
            model=model_name or settings.LLM_MODEL,
            model_provider=model_provider or settings.LLM_PROVIDER,
            api_key=settings.LLM_API_KEY,
        )
