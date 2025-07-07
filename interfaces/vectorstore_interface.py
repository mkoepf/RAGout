from typing import List, Tuple, Protocol


class VectorStoreInterface(Protocol):
    def add_texts(self, texts: List[str], metadatas: List[dict] = ...) -> None:
        """Add texts and metadata to the vectorstore."""
        ...

    def similarity_search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Tuple[str, dict]]:
        """Return k most similar texts with their metadata."""
        ...
