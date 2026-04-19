"""NewsAPI client for fetching real-time news articles."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlencode

from milestone2 import logger
from milestone2.config import NEWSAPI_KEY, NEWSAPI_BASE_URL

logger_news = logger.setup_logger("news", "news.log")


@dataclass
class NewsArticle:
    """Represents a news article from NewsAPI."""
    title: str
    description: str
    content: str
    url: str
    source: str
    published_at: str
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing."""
        return {
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "author": self.author,
        }

    @classmethod
    def from_api_response(cls, article_data: Dict[str, Any]) -> NewsArticle:
        """Create NewsArticle from NewsAPI response."""
        return cls(
            title=article_data.get("title", ""),
            description=article_data.get("description", ""),
            content=article_data.get("content", ""),
            url=article_data.get("url", ""),
            source=article_data.get("source", {}).get("name", ""),
            published_at=article_data.get("publishedAt", ""),
            author=article_data.get("author"),
        )


class NewsAPIClient:
    """Client for interacting with NewsAPI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NEWSAPI_KEY
        self.base_url = NEWSAPI_BASE_URL
        self.session = requests.Session()

        if not self.api_key:
            logger_news.warning("NewsAPI key not provided. Set NEWSAPI_KEY environment variable.")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to NewsAPI."""
        params["apiKey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger_news.error(f"NewsAPI request failed: {exc}")
            raise

    def get_top_headlines(
        self,
        country: str = "us",
        category: Optional[str] = None,
        sources: Optional[str] = None,
        q: Optional[str] = None,
        page_size: int = 20,
        page: int = 1,
    ) -> List[NewsArticle]:
        """Get top headlines from NewsAPI."""
        params = {
            "country": country,
            "pageSize": min(page_size, 100),  # NewsAPI max is 100
            "page": page,
        }

        if category:
            params["category"] = category
        if sources:
            params["sources"] = sources
        if q:
            params["q"] = q

        response = self._make_request("top-headlines", params)

        articles = []
        for article_data in response.get("articles", []):
            try:
                article = NewsArticle.from_api_response(article_data)
                articles.append(article)
            except Exception as exc:
                logger_news.warning(f"Failed to parse article: {exc}")
                continue

        logger_news.info(f"Retrieved {len(articles)} top headlines")
        return articles

    def get_everything(
        self,
        q: str,
        sources: Optional[str] = None,
        domains: Optional[str] = None,
        exclude_domains: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 20,
        page: int = 1,
    ) -> List[NewsArticle]:
        """Search all articles from NewsAPI."""
        params = {
            "q": q,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "page": page,
        }

        if sources:
            params["sources"] = sources
        if domains:
            params["domains"] = domains
        if exclude_domains:
            params["excludeDomains"] = exclude_domains
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        response = self._make_request("everything", params)

        articles = []
        for article_data in response.get("articles", []):
            try:
                article = NewsArticle.from_api_response(article_data)
                articles.append(article)
            except Exception as exc:
                logger_news.warning(f"Failed to parse article: {exc}")
                continue

        logger_news.info(f"Retrieved {len(articles)} articles for query: '{q}'")
        return articles

    def get_recent_news(
        self,
        query: str,
        days_back: int = 1,
        max_results: int = 50,
    ) -> List[NewsArticle]:
        """Get recent news articles for a query."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        articles = []
        page = 1

        while len(articles) < max_results:
            batch = self.get_everything(
                q=query,
                from_date=from_date,
                page_size=min(100, max_results - len(articles)),
                page=page,
            )

            if not batch:
                break

            articles.extend(batch)
            page += 1

            # NewsAPI free tier limits to 100 results per query
            if len(batch) < 100 or page > 5:
                break

        logger_news.info(f"Retrieved {len(articles)} recent articles for '{query}'")
        return articles[:max_results]

    def get_sources(
        self,
        category: Optional[str] = None,
        language: str = "en",
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get available news sources."""
        params = {"language": language}

        if category:
            params["category"] = category
        if country:
            params["country"] = country

        response = self._make_request("sources", params)
        sources = response.get("sources", [])

        logger_news.info(f"Retrieved {len(sources)} news sources")
        return sources