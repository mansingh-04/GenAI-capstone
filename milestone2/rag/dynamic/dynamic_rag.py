"""Dynamic RAG system for real-time news fact-checking using NewsAPI."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import chromadb
from chromadb.config import Settings

from logger import logger_rag
from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    DYNAMIC_RAG_TOP_K,
    NEWSAPI_KEY,
    USE_EXTERNAL_EMBEDDINGS,
    HUGGINGFACE_INFERENCE_URL,
    HUGGINGFACE_API_KEY,
)
from rag.dynamic.news_api_client import NewsAPIClient, NewsArticle


class DynamicRAG:
    """Dynamic RAG system for real-time news fact-checking."""

    def __init__(
        self,
        chroma_db_path: Optional[Path] = None,
        collection_name: str = "dynamic_news",
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = DYNAMIC_RAG_TOP_K,
    ):
        self.chroma_db_path = Path(chroma_db_path) if chroma_db_path else CHROMA_DB_PATH
        self.collection_name = collection_name
        self.top_k = top_k

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Memory-Efficient Initialization
        if USE_EXTERNAL_EMBEDDINGS:
            logger_rag.info("Using External Hugging Face Inference API for embeddings (RAM Optimization)")
            self.embedding_model = None
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model)
            except ImportError:
                logger_rag.error("sentence-transformers not found. Please install it or use external embeddings.")
                self.embedding_model = None

        # Initialize NewsAPI client
        self.news_client = NewsAPIClient()

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Dynamic news articles for fact-checking"}
        )

        logger_rag.info(
            f"DynamicRAG initialized: collection={collection_name}, "
            f"chroma_db={self.chroma_db_path}, model={embedding_model}"
        )

    def _generate_article_id(self, article: NewsArticle) -> str:
        """Generate unique ID for article based on URL and content hash."""
        content_hash = hashlib.md5(
            f"{article.title}{article.description}{article.url}".encode()
        ).hexdigest()[:8]
        return f"news_{content_hash}"

    def _prepare_article_for_indexing(self, article: NewsArticle) -> Dict[str, Any]:
        """Prepare article data for ChromaDB indexing."""
        article_id = self._generate_article_id(article)

        # Combine title and description for embedding
        text_content = f"{article.title}. {article.description or ''}"

        # Create metadata
        metadata = {
            "title": article.title,
            "description": article.description or "",
            "url": article.url,
            "source": article.source,
            "published_at": article.published_at,
            "author": article.author or "",
            "content_type": "news_article",
            "indexed_at": datetime.now().isoformat(),
        }

        return {
            "id": article_id,
            "text": text_content,
            "metadata": metadata,
        }

    def index_news_articles(
        self,
        articles: List[NewsArticle],
        reset: bool = False,
    ) -> None:
        """Index news articles into ChromaDB."""
        if reset:
            logger_rag.info(f"Resetting existing collection: {self.collection_name}")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Dynamic news articles for fact-checking"}
            )

        if not articles:
            logger_rag.info("No articles to index")
            return

        # Prepare data for indexing
        ids = []
        texts = []
        metadatas = []

        for article in articles:
            prepared = self._prepare_article_for_indexing(article)
            ids.append(prepared["id"])
            texts.append(prepared["text"])
            metadatas.append(prepared["metadata"])

        # Generate embeddings
        if USE_EXTERNAL_EMBEDDINGS:
            try:
                response = requests.post(
                    HUGGINGFACE_INFERENCE_URL,
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=20
                )
                response.raise_for_status()
                embeddings = response.json()
                if isinstance(embeddings, list) and len(embeddings) > 0 and not isinstance(embeddings[0], list):
                    embeddings = [embeddings]
            except Exception as e:
                logger_rag.error(f"External embedding API failed: {e}")
                embeddings = [[0.0] * 384] * len(texts)
        else:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized and external embeddings disabled.")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

        # Index in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=texts[i:end_idx],
            )

        logger_rag.info(f"Indexed {len(articles)} news articles into ChromaDB")

    def fetch_and_index_recent_news(
        self,
        query: str,
        days_back: int = 1,
        max_articles: int = 50,
        reset: bool = False,
    ) -> int:
        """Fetch recent news and index them."""
        logger_rag.info(f"Fetching recent news for query: '{query}' ({days_back} days back)")

        try:
            articles = self.news_client.get_recent_news(
                query=query,
                days_back=days_back,
                max_results=max_articles,
            )

            if articles:
                self.index_news_articles(articles, reset=reset)
                return len(articles)
            else:
                logger_rag.warning(f"No articles found for query: '{query}'")
                return 0

        except Exception as exc:
            logger_rag.error(f"Failed to fetch and index news: {exc}")
            return 0

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query the dynamic news collection."""
        if top_k is None:
            top_k = self.top_k

        logger_rag.info(f"Querying Dynamic RAG: '{query_text}' (top_k={top_k})")

        # Generate query embedding
        if USE_EXTERNAL_EMBEDDINGS:
            try:
                response = requests.post(
                    HUGGINGFACE_INFERENCE_URL,
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json={"inputs": [query_text], "options": {"wait_for_model": True}},
                    timeout=20
                )
                response.raise_for_status()
                query_embedding = response.json()
                if isinstance(query_embedding, list) and not isinstance(query_embedding[0], list):
                    query_embedding = [query_embedding]
            except Exception as e:
                logger_rag.error(f"External embedding API failed: {e}")
                query_embedding = [[0.0] * 384]
        else:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized and external embeddings disabled.")
            query_embedding = [self.embedding_model.encode([query_text])[0].tolist()]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "claim": doc,
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "distance": results["distances"][0][i],
                }

                if include_metadata and results["metadatas"][0][i]:
                    result["metadata"] = results["metadatas"][0][i]

                formatted_results.append(result)

        logger_rag.info(f"Dynamic RAG returned {len(formatted_results)} results")
        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "count": count,
            "description": "Dynamic news articles for real-time fact-checking",
        }

    def search_similar_articles(
        self,
        article_text: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find articles similar to the given article text."""
        return self.query(article_text, top_k=top_k)

    def get_top_headlines_and_index(
        self,
        category: Optional[str] = None,
        country: str = "us",
        max_articles: int = 20,
        reset: bool = False,
    ) -> int:
        """Fetch top headlines and index them."""
        logger_rag.info(f"Fetching top headlines (category: {category}, country: {country})")

        try:
            articles = self.news_client.get_top_headlines(
                category=category,
                country=country,
                page_size=max_articles,
            )

            if articles:
                self.index_news_articles(articles, reset=reset)
                return len(articles)
            else:
                logger_rag.warning("No top headlines found")
                return 0

        except Exception as exc:
            logger_rag.error(f"Failed to fetch top headlines: {exc}")
            return 0