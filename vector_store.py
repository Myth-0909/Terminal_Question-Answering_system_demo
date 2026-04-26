"""Milvus Lite vector store operations."""

from __future__ import annotations

import os
import random
import time
from typing import Callable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from pymilvus.exceptions import ConnectionConfigException
from sklearn.feature_extraction.text import HashingVectorizer

from config import AppConfig
from data_generator import profile_to_text


class LocalHashEmbeddings(Embeddings):
    """Offline fallback embeddings to keep demo runnable without model download."""

    def __init__(self, dimensions: int = 768) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            analyzer="char",
            ngram_range=(1, 2),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].tolist()


class MilvusStoreManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        os.environ.setdefault("HF_ENDPOINT", config.hf_endpoint)
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(config.hf_download_timeout))
        if config.embedding_backend == "local_hash":
            self.embeddings = LocalHashEmbeddings(dimensions=768)
        elif config.embedding_backend == "hf":
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                raise RuntimeError(f"HuggingFace embedding 初始化失败：{exc}") from exc
        else:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                print(f"[WARN] HuggingFace embedding init failed, fallback to local hash embeddings: {exc}")
                self.embeddings = LocalHashEmbeddings(dimensions=768)
        self.client = self._build_client()
        self._ensure_search_ready()

    def _build_client(self) -> MilvusClient:
        self.config.milvus_path.parent.mkdir(parents=True, exist_ok=True)
        last_exc: Exception | None = None
        for _ in range(5):
            try:
                return MilvusClient(uri=str(self.config.milvus_path))
            except ConnectionConfigException as exc:
                last_exc = exc
                time.sleep(0.4)
        raise RuntimeError(
            "Milvus 本地数据库正在被其他进程占用，请先关闭其他 demo 终端后重试。"
        ) from last_exc

    def _ensure_collection(self, dim: int) -> None:
        if self.client.has_collection(self.config.collection_name):
            return
        self.client.create_collection(
            collection_name=self.config.collection_name,
            dimension=dim,
            metric_type="COSINE",
            auto_id=True,
            id_type="int",
            enable_dynamic_field=True,
        )
        self._create_vector_index()
        self.client.load_collection(self.config.collection_name)

    def _create_vector_index(self) -> None:
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_idx",
            params={"nlist": 128},
        )
        self.client.create_index(
            collection_name=self.config.collection_name,
            index_params=index_params,
        )

    def _ensure_search_ready(self) -> None:
        if not self.client.has_collection(self.config.collection_name):
            return
        indexes = self.client.list_indexes(self.config.collection_name)
        if not indexes:
            self._create_vector_index()
        self.client.load_collection(self.config.collection_name)

    def has_indexed_data(self) -> bool:
        if not self.client.has_collection(self.config.collection_name):
            return False
        stats = self.client.get_collection_stats(self.config.collection_name)
        row_count = int(stats.get("row_count", 0))
        return row_count > 0

    def get_row_count(self) -> int:
        if not self.client.has_collection(self.config.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.config.collection_name)
        return int(stats.get("row_count", 0))

    def reset_collection(self) -> None:
        if self.client.has_collection(self.config.collection_name):
            self.client.drop_collection(self.config.collection_name)

    def ingest_profiles(
        self,
        profiles: Sequence[dict],
        batch_size: int,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> None:
        docs: list[Document] = [Document(page_content=profile_to_text(p), metadata=p) for p in profiles]
        total = len(docs)
        for offset in range(0, total, batch_size):
            chunk = docs[offset : offset + batch_size]
            texts = [item.page_content for item in chunk]
            vectors = self.embeddings.embed_documents(texts)
            self._ensure_collection(dim=len(vectors[0]))
            rows = []
            for doc, vec in zip(chunk, vectors):
                rows.append(
                    {
                        "vector": vec,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )
            self.client.insert(collection_name=self.config.collection_name, data=rows)
            if progress_cb is not None:
                progress_cb(min(offset + batch_size, total), total)

    def similarity_search(self, query: str, top_k: int = 5) -> list[Document]:
        self._ensure_search_ready()
        query_vec = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["content", "metadata"],
        )
        results: list[Document] = []
        for item in hits[0]:
            entity = item.get("entity", {})
            results.append(
                Document(
                    page_content=entity.get("content", ""),
                    metadata=entity.get("metadata", {}),
                )
            )
        return results

    def count_profiles_by_surname(self, surname: str) -> int:
        """Count all records whose metadata.name starts with a surname."""
        if not surname:
            return 0
        self._ensure_search_ready()
        iterator = self.client.query_iterator(
            collection_name=self.config.collection_name,
            batch_size=500,
            limit=-1,
            filter="",
            output_fields=["metadata"],
        )
        count = 0
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    metadata = row.get("metadata", {}) or {}
                    name = str(metadata.get("name", "")).strip()
                    if name.startswith(surname):
                        count += 1
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        return count

    def find_profiles_by_name(self, name: str, max_results: int = 3) -> list[dict]:
        """Find profiles by exact name first, then fuzzy contains match."""
        target = (name or "").strip()
        if not target:
            return []
        self._ensure_search_ready()
        iterator = self.client.query_iterator(
            collection_name=self.config.collection_name,
            batch_size=500,
            limit=-1,
            filter="",
            output_fields=["metadata"],
        )
        exact_matches: list[dict] = []
        fuzzy_matches: list[dict] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    metadata = row.get("metadata", {}) or {}
                    row_name = str(metadata.get("name", "")).strip()
                    if not row_name:
                        continue
                    if row_name == target:
                        exact_matches.append(metadata)
                    elif target in row_name:
                        fuzzy_matches.append(metadata)
        finally:
            if hasattr(iterator, "close"):
                iterator.close()

        ranked = exact_matches + fuzzy_matches
        return ranked[:max_results]

    def get_random_profiles(self, max_results: int = 1) -> list[dict]:
        self._ensure_search_ready()
        iterator = self.client.query_iterator(
            collection_name=self.config.collection_name,
            batch_size=500,
            limit=-1,
            filter="",
            output_fields=["metadata"],
        )
        all_profiles: list[dict] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    metadata = row.get("metadata", {}) or {}
                    if metadata:
                        all_profiles.append(metadata)
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        if not all_profiles:
            return []
        sample_size = min(max_results, len(all_profiles))
        return random.sample(all_profiles, k=sample_size)
