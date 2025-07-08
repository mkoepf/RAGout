import argparse
import os
import sys

from infrastructure.embedding_ollama import OllamaEmbedder
from infrastructure.vectorstore_faiss import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query the arXiv FAISS vector store with a natural language "
            "question."
        )
    )
    parser.add_argument("query", type=str, help="Your search query.")
    parser.add_argument(
        "--index-path",
        type=str,
        default="arxiv_faiss.index",
        help="Path to the FAISS index file (default: arxiv_faiss.index)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="arxiv_metadata.pkl",
        help="Path to the metadata file (default: arxiv_metadata.pkl)",
    )
    args = parser.parse_args()

    print(
        f"Loading vector store from {args.index_path} "
        f"and {args.metadata_path}..."
    )
    vectorstore = FaissVectorStore(
        dim=1, index_path=args.index_path, metadata_path=args.metadata_path
    )
    vectorstore.load()

    print("Initializing embedder...")
    embedder = OllamaEmbedder()
    print(f"Embedding query: {args.query}")
    query_embedding = embedder.embed_query(args.query)
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
