import os
import time
import logging
import warnings

import numpy as np
import pytest
from testcontainers.core.container import DockerContainer

from infrastructure.embedding_ollama import OllamaEmbedder
from infrastructure.vectorstore_faiss import FaissVectorStore

# Suppress numpy/faiss deprecation warnings for cleaner test output
# These warnings are caused by internal usage in faiss/numpy and do not affect
# the correctness or future compatibility of our own code. Once faiss updates
# their codebase, these suppressions can be removed.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_PORT = 11434

# Enable debug logging for testcontainers
logging.basicConfig(level=logging.INFO)
logging.getLogger("testcontainers").setLevel(logging.DEBUG)

# Register the custom pytest mark to avoid warnings
# Add this to the top-level conftest.py or pytest.ini in your project root:
# [pytest]
# markers = integration: mark a test as an integration test.


@pytest.mark.integration
def test_embedding_ollama_and_vectorstore_faiss_with_testcontainers():
    """
    Integration test: Use OllamaEmbedder and FaissVectorStore with Ollama
    running in a Docker container via testcontainers.
    Ensures the required model is pulled and available.
    """
    print("Starting Ollama container...")
    with DockerContainer("ollama/ollama:latest").with_exposed_ports(
        OLLAMA_PORT
    ).with_command(None) as ollama:
        print("Container started. Showing logs (first 20 lines):")
        logs = ollama.get_logs()
        if isinstance(logs, tuple):
            logs = (logs[0] or b'').decode() + (logs[1] or b'').decode()
        for i, line in enumerate(logs.splitlines()):
            print(line)
            if i > 20:
                print("...log output truncated...")
                break
        host = ollama.get_container_host_ip()
        port = ollama.get_exposed_port(OLLAMA_PORT)
        base_url = f"http://{host}:{port}"

        print("Waiting for Ollama to be ready...")
        for _ in range(30):
            try:
                import requests

                r = requests.get(f"{base_url}/api/tags")
                if r.status_code == 200:
                    print("Ollama is ready.")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Ollama did not start in time")

        print(f"Pulling model '{OLLAMA_MODEL}'...")
        import requests

        pull_resp = requests.post(
            f"{base_url}/api/pull", json={"name": OLLAMA_MODEL}
        )
        assert pull_resp.status_code == 200 or pull_resp.status_code == 201
        print("Waiting for model to be available...")
        for _ in range(30):
            tags = requests.get(f"{base_url}/api/tags").json()
            if any(OLLAMA_MODEL in m["name"] for m in tags.get("models", [])):
                print(f"Model '{OLLAMA_MODEL}' is available.")
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Model {OLLAMA_MODEL} not available in Ollama")

        print("Running embedding and vectorstore integration test...")
        embedder = OllamaEmbedder(model_name=OLLAMA_MODEL, base_url=base_url)
        sample_embedding = embedder.embed_query("test")
        dim = len(sample_embedding)
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "faiss.index")
            metadata_path = os.path.join(tmpdir, "metadata.pkl")
            vectorstore = FaissVectorStore(
                dim=dim, index_path=index_path, metadata_path=metadata_path
            )
            texts = ["The quick brown fox.", "Jumps over the lazy dog."]
            metadatas = [{"id": 1}, {"id": 2}]
            embeddings = [embedder.embed_query(text) for text in texts]
            vectorstore.index.add(np.array(embeddings).astype("float32"))
            vectorstore.texts.extend(texts)
            vectorstore.metadatas.extend(metadatas)
            vectorstore._save()
            query = "A fast brown animal."
            query_embedding = embedder.embed_query(query)
            results = vectorstore.similarity_search(query_embedding, k=2)
            assert len(results) == 2
            assert any("fox" in text for text, _ in results)
            assert any("dog" in text for text, _ in results)
            for text, meta in results:
                print(f"Text: {text}, Metadata: {meta}")
