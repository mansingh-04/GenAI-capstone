"""Static RAG implementation using LIAR dataset and ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from logger import logger_rag
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_MODEL,
    STATIC_RAG_TOP_K,
)
from rag.static.liar_dataset_loader import LIARDatasetLoader


class StaticRAG:
    def __init__(
        self,
        chroma_db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.chroma_db_path = Path(chroma_db_path) if chroma_db_path else CHROMA_DB_PATH
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.distance_metric = distance_metric or CHROMA_DISTANCE_METRIC
        self.top_k = top_k or STATIC_RAG_TOP_K

        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"distance_metric": self.distance_metric},
        )

        logger_rag.info(
            f"StaticRAG initialized: collection={self.collection_name}, "
            f"chroma_db={self.chroma_db_path}, model={self.embedding_model_name}"
        )

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedding_model.encode(list(texts), show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]

    def index_fact_checks(
        self,
        facts: List[Dict[str, object]],
        reset: bool = False,
    ) -> None:
        if reset and self.collection:
            logger_rag.info(f"Resetting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"distance_metric": self.distance_metric},
            )

        ids = [fact["id"] for fact in facts]
        documents = [fact["claim"] for fact in facts]
        metadatas = [fact.get("metadata", {}) for fact in facts]
        embeddings = self._embed(documents)

        logger_rag.info(f"Indexing {len(documents)} fact-check claims into ChromaDB")
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger_rag.info("Static RAG indexing completed")

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, object]]:
        top_k = top_k or self.top_k
        logger_rag.info(f"Querying Static RAG: '{query_text[:80]}' (top_k={top_k})")

        query_embeddings = self._embed(query_text)
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: List[Dict[str, object]] = []
        for claim, metadata, distance in zip(documents, metadatas, distances):
            output.append(
                {
                    "claim": claim,
                    "metadata": metadata,
                    "distance": float(distance),
                    "score": 1.0 - float(distance),
                }
            )

        logger_rag.info(f"Static RAG returned {len(output)} results")
        return output

    def get_collection_info(self) -> Dict[str, object]:
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": str(self.chroma_db_path),
            "top_k": self.top_k,
            "distance_metric": self.distance_metric,
        }

    def build_from_local_dataset(
        self,
        dataset_path: Optional[Path] = None,
        reset: bool = False,
    ) -> None:
        loader = LIARDatasetLoader(dataset_path=dataset_path)
        facts = loader.load_dataset(dataset_path=dataset_path)
        self.index_fact_checks(facts, reset=reset)
