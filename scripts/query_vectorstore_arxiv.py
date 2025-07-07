import os
import sys

from infrastructure.embedding_ollama import OllamaEmbedder
from infrastructure.vectorstore_faiss import FaissVectorStore


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/query_vectorstore_arxiv.py '<your query>'"
            "[index_path] [metadata_path]"
        )
        sys.exit(1)
    query = sys.argv[1]
    index_path = sys.argv[2] if len(sys.argv) > 2 else "arxiv_faiss.index"
    metadata_path = sys.argv[3] if len(sys.argv) > 3 else "arxiv_metadata.pkl"

    print(f"Loading vector store from {index_path} and {metadata_path}...")
    # Load the vectorstore
    vectorstore = FaissVectorStore(
        dim=1, index_path=index_path, metadata_path=metadata_path
    )
    vectorstore.load()

    print("Initializing embedder...")
    embedder = OllamaEmbedder()
    print(f"Embedding query: {query}")
    query_embedding = embedder.embed_query(query)
    print("Running similarity search...")
    results = vectorstore.similarity_search(query_embedding, k=5)
    print("Top 5 results:")
    for i, (text, meta) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {meta.get('title', 'N/A')}")
        print(f"ID: {meta.get('id', 'N/A')}")
        print(f"Abstract: {text[:500]}{'...' if len(text) > 500 else ''}")


if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    main()
