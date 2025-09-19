from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingsManager:
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self._model_name = model_name
        self._embeddings = None  # lazy initialization

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            print("Loading embeddings model...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._embeddings
