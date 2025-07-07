from unittest.mock import MagicMock, patch

import pytest

from infrastructure.embedding_ollama import OllamaEmbedder


def test_embed_documents_success():
    """
    Test that embed_documents returns correct embeddings for a list of texts.
    Mocks the Ollama API response to return a fixed embedding.
    """
    embedder = OllamaEmbedder()
    texts = ["hello", "world"]
    fake_embedding = [0.1, 0.2, 0.3]
    with patch("infrastructure.embedding_ollama.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": fake_embedding}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        result = embedder.embed_documents(texts)
        assert result == [fake_embedding, fake_embedding]
        assert mock_post.call_count == 2


def test_embed_query_success():
    """
    Test that embed_query returns the correct embedding for a single query.
    Mocks the Ollama API response to return a fixed embedding.
    """
    embedder = OllamaEmbedder()
    query = "test query"
    fake_embedding = [0.4, 0.5, 0.6]
    with patch("infrastructure.embedding_ollama.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": fake_embedding}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        result = embedder.embed_query(query)
        assert result == fake_embedding
        mock_post.assert_called_once()


def test_embed_documents_http_error():
    """
    Test that embed_documents raises an exception when the Ollama API returns
    an error.
    """
    embedder = OllamaEmbedder()
    texts = ["fail"]
    with patch("infrastructure.embedding_ollama.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP error")
        mock_post.return_value = mock_response
        with pytest.raises(Exception) as excinfo:
            embedder.embed_documents(texts)
        assert "HTTP error" in str(excinfo.value)


def test_embed_query_http_error():
    """
    Test that embed_query raises an exception when the Ollama API returns an
    error.
    """
    embedder = OllamaEmbedder()
    query = "fail query"
    with patch("infrastructure.embedding_ollama.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP error")
        mock_post.return_value = mock_response
        with pytest.raises(Exception) as excinfo:
            embedder.embed_query(query)
        assert "HTTP error" in str(excinfo.value)
