import numpy as np
from datasets import load_dataset

from infrastructure.embedding_ollama import OllamaEmbedder
from infrastructure.vectorstore_faiss import FaissVectorStore


def build_arxiv_vectorstore(
    out_index_path: str = "arxiv_faiss.index",
    out_metadata_path: str = "arxiv_metadata.pkl",
    num_papers: int = 1000,
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
):
    """
    Download papers from the Hugging Face arxiv-abstracts-2021 dataset,
    embed their abstracts, and store them in a FaissVectorStore.

    Args:
        out_index_path (str): Path to save the FAISS index file.
        out_metadata_path (str): Path to save the metadata file.
        num_papers (int): Number of papers to process from the dataset.
        model_name (str): Name of the embedding model to use.
        base_url (str): Base URL of the Ollama server.
    """
    print("Loading dataset from Hugging Face...")
    ds = load_dataset(
        "gfissore/arxiv-abstracts-2021", split="train[:{}]".format(num_papers)
    )
    abstracts = [item["abstract"] for item in ds]
    metadatas = [{"id": item["id"], "title": item["title"]} for item in ds]

    print("Initializing embedder...")
    embedder = OllamaEmbedder(model_name=model_name, base_url=base_url)
    print("Embedding abstracts...")
    embeddings = [embedder.embed_query(text) for text in abstracts]
    dim = len(embeddings[0])

    print(f"Embedding dimension: {dim}")

    print("Building vector store...")
    vectorstore = FaissVectorStore(
        dim=dim, index_path=out_index_path, metadata_path=out_metadata_path
    )
    vectorstore.index.add(
        np.array(embeddings).astype("float32")  # type: ignore
    )
    vectorstore.texts.extend(abstracts)
    vectorstore.metadatas.extend(metadatas)
    vectorstore._save()
    print(f"Saved vector store to {out_index_path} and {out_metadata_path}")


if __name__ == "__main__":
    """
    Entry point for building the arxiv vector store. Downloads, embeds, and
    saves the data.
    """
    build_arxiv_vectorstore()
