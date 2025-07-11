import argparse
from typing import Any, Dict, List, Tuple

from fastmcp import FastMCP

from infrastructure.embedding_ollama import OllamaEmbedder
from infrastructure.vectorstore_faiss import FaissVectorStore
from interfaces.embedder_interface import EmbedderInterface
from interfaces.vectorstore_interface import VectorStoreInterface

mcp: FastMCP[Any] = FastMCP("MCP demo server")


class RagMCP:
    """
    RagMCP is a FastMCP server that provides a tool to search the arXiv
    vector store for relevant papers based on user queries.
    """

    embedder: EmbedderInterface
    vectorstore: VectorStoreInterface

    def __init__(
        self, embedder: EmbedderInterface, vectorstore: VectorStoreInterface
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    @mcp.tool
    def search_arxiv(self, query: str) -> List[Tuple[str, Dict]]:
        """
        Search the arXiv vector store for papers relevant to the given query.

        This tool embeds the input query using the OllamaEmbedder, performs a
        similarity search in the FAISS-based vector store of arXiv abstracts, and
        returns the top 5 most relevant results.

        Args:
            query (str): The user's search query.

        Returns:
            List[Tuple[str, Dict]]: A list of tuples, each containing the abstract
            text and its metadata dictionary for the top matching papers.
        """
        print(f"Embedding query: {query}")
        query_embedding = embedder.embed_query(query)
        print("Running similarity search...")
        results = vectorstore.similarity_search(query_embedding, k=5)
        # print("Top 5 results:")
        # for i, (text, meta) in enumerate(results, 1):
        #     print(f"\nResult {i}:")
        #     print(f"Title: {meta.get('title', 'N/A')}")
        #     print(f"ID: {meta.get('id', 'N/A')}")
        #     print(f"Abstract: {text[:500]}{'...' if len(text) > 500 else ''}")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the FastMCP server " "in stdio or sse mode."
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="Server mode: stdio (default) or sse",
    )
    args = parser.parse_args()

    embedder = OllamaEmbedder()

    index_path = "arxiv_faiss.index"
    metadata_path = "arxiv_metadata.pkl"

    vectorstore = FaissVectorStore(
        dim=768, index_path=index_path, metadata_path=metadata_path
    )
    vectorstore.load()

    server = RagMCP(embedder, vectorstore)

    print("Starting the FastMCP server...")
    if args.mode == "sse":
        mcp.run("sse")
    else:
        mcp.run("stdio")
