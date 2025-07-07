import pickle
from typing import List, Optional, Tuple

import faiss
import numpy as np


class FaissVectorStore:
    """
    FaissVectorStore manages storage and retrieval of text embeddings using
    FAISS.

    Args:
        dim (int): Dimensionality of the embeddings.
        index_path (str): Path to save/load the FAISS index file.
        metadata_path (str): Path to save/load the metadata file.
    """

    def __init__(
        self,
        dim: int,
        index_path: str = "index/faiss.index",
        metadata_path: str = "index/metadata.pkl",
    ):
        """
        Initialize the FaissVectorStore with embedding dimension, index path,
        and metadata path.
        """
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """
        Add texts and their metadata to the vector store.

        Args:
            texts (List[str]): List of texts to add.
            metadatas (Optional[List[dict]]): List of metadata dicts for each
            text.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        embeddings = self._embed_texts(texts)  # Placeholder; replace if needed
        self.index.add(np.array(embeddings).astype("float32"))  # type: ignore
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self._save()

    def similarity_search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Tuple[str, dict]]:
        """
        Perform a similarity search for the given query embedding.

        Args:
            query_embedding (List[float]): Embedding vector for the query.
            k (int): Number of top results to return.
        Returns:
            List[Tuple[str, dict]]: List of (text, metadata) tuples for the top
            matches.
        """
        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, k)  # type: ignore
        return [(self.texts[i], self.metadatas[i]) for i in indices[0]]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Placeholder for embedding texts. Should be replaced with an external
        embedder.

        Args:
            texts (List[str]): List of texts to embed.
        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        raise NotImplementedError(
            "Use embedder externally; this is a placeholder."
        )

    def _save(self):
        """
        Save the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump((self.texts, self.metadatas), f)

    def load(self):
        """
        Load the FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.texts, self.metadatas = pickle.load(f)
