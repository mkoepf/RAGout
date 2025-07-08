# RAGout

RAGout is a modular implementation of RAG (Retrieval-Augmented Generation).

In its current form it uses Ollama for embedding and FAISS for vector storage,
but it's extensible to support various LLM APIs and vector stores.

For demonstration and testing, it includes scripts to populate a vector store
with arXiv abstracts, which are taken from a subset of the
[gfissore/arxiv-abstracts-2021
dataset](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) from
Hugging Face.

## Features
- Embedding with Ollama (local LLMs)
- Fast vector search with FAISS
- Scripts to populate/query vector store from real data
- Unit and integration tests (with testcontainers)
- Pre-commit hooks for lint, type, and test checks
- GitHub Actions CI

## Quickstart

### 1. Install dependencies
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### 2. Build the vector store
```sh
python scripts/populate_vectorstore_arxiv.py
```

### 3. Query the vector store
```sh
python scripts/query_vectorstore_arxiv.py "What is quantum computing?"
```

## Testing
- **Fast tests:** `pytest -m "not integration"`
- **Integration tests:** `pytest -m "integration"`

## Lint & Type Check
```sh
ruff check .
python -m mypy . --follow-untyped-imports
```

## Pre-commit
Run all checks manually:
```sh
pre-commit run --all-files
```

## CI
See `.github/workflows/ci.yml` for automated lint, type, and test runs.

---

**Project structure:**
- `infrastructure/` — Embedding & vector store implementations
- `interfaces/` — Abstract interfaces
- `scripts/` — Data and query scripts
- `tests/` — Unit & integration tests

---

MIT License
