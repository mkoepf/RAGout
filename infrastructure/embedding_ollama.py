import requests
from typing import List


class OllamaEmbedder:
    """
    OllamaEmbedder provides embedding functionality using an Ollama server.

    Args:
        model_name (str): Name of the embedding model to use. Defaults to
        'nomic-embed-text'.  base_url (str): Base URL of the Ollama server.
        Defaults to 'http://localhost:11434'.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the OllamaEmbedder with model name and base URL.
        """
        self.model = model_name
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): List of input texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors for each input text.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            query (str): The input query string.

        Returns:
            List[float]: The embedding vector for the query.
        """
        return self._embed(query)

    def _embed(self, text: str) -> List[float]:
        """
        Internal method to call the Ollama API and get the embedding for a
        single text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector for the input text.

        Raises:
            requests.HTTPError: If the Ollama API returns a non-200 status
            code.
        """
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
