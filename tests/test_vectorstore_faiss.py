import os
import tempfile

import numpy as np
import pytest

from infrastructure.vectorstore_faiss import FaissVectorStore


class DummyFaissIndex:
    """
    DummyFaissIndex is a mock class to simulate the FAISS index for testing.
    """

    def __init__(self):
        self.added = []

    def add(self, arr):
        """
        Mock add method to store added arrays.
        """
        self.added.append(arr)

    def search(self, query, k):
        """
        Mock search method to return dummy distances and unique indices.
        """
        return np.zeros((1, k)), np.array([list(range(k))])


def test_add_texts_and_similarity_search(monkeypatch):
    """
    Test adding texts and metadata, and performing similarity search.
    Uses monkeypatching to mock FAISS and embedding behavior.
    """
    # Use temp files for index and metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "faiss.index")
        metadata_path = os.path.join(tmpdir, "metadata.pkl")
        store = FaissVectorStore(
            dim=3, index_path=index_path, metadata_path=metadata_path
        )
        # Patch faiss.IndexFlatL2 and faiss.write_index/read_index
        monkeypatch.setattr(store, "index", DummyFaissIndex())
        monkeypatch.setattr("faiss.write_index", lambda index, path: None)
        monkeypatch.setattr("faiss.read_index", lambda path: DummyFaissIndex())
        # Patch _embed_texts to return fixed embeddings
        store._embed_texts = lambda texts: [[1.0, 2.0, 3.0] for _ in texts]
        texts = ["foo", "bar"]
        metadatas = [{"id": 1}, {"id": 2}]
        store.add_texts(texts, metadatas)
        assert store.texts == texts
        assert store.metadatas == metadatas
        # Test similarity_search
        result = store.similarity_search([1.0, 2.0, 3.0], k=2)
        assert result == [("foo", {"id": 1}), ("bar", {"id": 2})]


def test_add_texts_without_metadatas(monkeypatch):
    """
    Test adding texts without providing metadata. Should default to empty
    dicts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "faiss.index")
        metadata_path = os.path.join(tmpdir, "metadata.pkl")
        store = FaissVectorStore(
            dim=2, index_path=index_path, metadata_path=metadata_path
        )
        monkeypatch.setattr(store, "index", DummyFaissIndex())
        monkeypatch.setattr("faiss.write_index", lambda index, path: None)
        store._embed_texts = lambda texts: [[0.0, 0.0] for _ in texts]
        texts = ["baz"]
        store.add_texts(texts)
        assert store.texts == ["baz"]
        assert store.metadatas == [{}]


def test_embed_texts_not_implemented():
    """
    Test that calling _embed_texts raises NotImplementedError as expected.
    """
    store = FaissVectorStore(dim=2)
    with pytest.raises(NotImplementedError):
        store._embed_texts(["test"])


def test_save_and_load(monkeypatch):
    """
    Test saving and loading the FAISS index and metadata using temp files.
    """
    # Use temp files for index and metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "faiss.index")
        metadata_path = os.path.join(tmpdir, "metadata.pkl")
        store = FaissVectorStore(
            dim=2, index_path=index_path, metadata_path=metadata_path
        )
        monkeypatch.setattr(store, "index", DummyFaissIndex())
        monkeypatch.setattr("faiss.write_index", lambda index, path: None)
        monkeypatch.setattr("faiss.read_index", lambda path: DummyFaissIndex())
        store.texts = ["a"]
        store.metadatas = [{"foo": "bar"}]
        # Actually write metadata to file
        store._save()
        # Create a new store and load
        new_store = FaissVectorStore(
            dim=2, index_path=index_path, metadata_path=metadata_path
        )
        monkeypatch.setattr(new_store, "index", DummyFaissIndex())
        monkeypatch.setattr("faiss.read_index", lambda path: DummyFaissIndex())
        new_store.load()
        assert new_store.texts == ["a"]
        assert new_store.metadatas == [{"foo": "bar"}]
