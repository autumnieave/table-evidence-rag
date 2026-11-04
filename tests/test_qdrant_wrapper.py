"""Qdrant collection schema fail-fast 单测（不连接真实 Qdrant）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from vectorstore.qdrant_wrapper import QdrantClientWrapper, QdrantCollectionSchemaError


COLLECTION = "test_collection"


def _wrapper(client, *, dimension: int = 1536, distance: str = "COSINE") -> QdrantClientWrapper:
    wrapper = object.__new__(QdrantClientWrapper)
    wrapper.config = SimpleNamespace(
        QDRANT_COLLECTION_NAME=COLLECTION,
        VECTOR_DIMENSION=dimension,
        DISTANCE_METRIC=distance,
    )
    wrapper.models = models
    wrapper.client = client
    return wrapper


class _FakeClient:
    def __init__(self, vectors=None, *, exists: bool = True):
        self.vectors = vectors
        self.exists = exists
        self.created = None

    def get_collections(self):
        collections = [SimpleNamespace(name=COLLECTION)] if self.exists else []
        return SimpleNamespace(collections=collections)

    def get_collection(self, name):
        assert name == COLLECTION
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors))
        )

    def create_collection(self, **kwargs):
        self.created = kwargs
        return True


def test_existing_collection_matching_schema_passes():
    client = _FakeClient(models.VectorParams(size=1536, distance=models.Distance.COSINE))
    _wrapper(client)._ensure_collection_exists()


def test_named_vector_matching_schema_passes():
    client = _FakeClient({"text": models.VectorParams(size=1536, distance=models.Distance.COSINE)})
    _wrapper(client)._ensure_collection_exists()


def test_dimension_mismatch_fails_fast():
    client = _FakeClient(models.VectorParams(size=1024, distance=models.Distance.COSINE))
    with pytest.raises(QdrantCollectionSchemaError, match="维度不匹配"):
        _wrapper(client)._ensure_collection_exists()


def test_distance_mismatch_fails_fast():
    client = _FakeClient(models.VectorParams(size=1536, distance=models.Distance.DOT))
    with pytest.raises(QdrantCollectionSchemaError, match="距离度量不匹配"):
        _wrapper(client)._ensure_collection_exists()


def test_named_vector_mismatch_fails_fast():
    client = _FakeClient({"text": models.VectorParams(size=1024, distance=models.Distance.COSINE)})
    with pytest.raises(QdrantCollectionSchemaError, match="text.*维度不匹配"):
        _wrapper(client)._ensure_collection_exists()


def test_creation_uses_configured_dimension_and_distance():
    client = _FakeClient(exists=False)
    _wrapper(client, dimension=1024, distance="EUCLIDEAN")._ensure_collection_exists()
    assert client.created is not None
    params = client.created["vectors_config"]
    assert params.size == 1024
    assert params.distance == models.Distance.EUCLID
