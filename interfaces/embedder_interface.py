from typing import List, Protocol


class EmbedderInterface(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of documents into embedding vectors."""
        ...

    def embed_query(self, query: str) -> List[float]:
        """Convert a query string into a single embedding vector."""
        ...
