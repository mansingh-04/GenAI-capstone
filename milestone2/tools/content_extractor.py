"""
Content Extraction Module
Extracts and cleans article text from URLs or raw text input
"""

import re
import string
from typing import Dict, Tuple
from urllib.parse import urlparse

from newspaper import Article
import requests
from bs4 import BeautifulSoup

from milestone2.logger import logger_tools
from milestone2.constants import ERROR_EXTRACTION_FAILED


class ContentExtractor:
    """Extract and clean article content from URLs or raw text"""

    # User agent for web requests (avoid being blocked)
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )

    # Valid URL schemes
    VALID_SCHEMES = ("http", "https")

    # Minimum text length for valid extraction
    MIN_TEXT_LENGTH = 50

    def __init__(self):
        """Initialize content extractor"""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        logger_tools.debug("ContentExtractor initialized")

    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid

        Args:
            url: URL string to validate

        Returns:
            True if URL is valid, False otherwise
        """
        try:
            if not url or not isinstance(url, str):
                return False

            url = url.strip()

            # Check URL format
            if not url.startswith(("http://", "https://")):
                return False

            # Parse URL
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in self.VALID_SCHEMES:
                return False

            # Check netloc (domain)
            if not parsed.netloc:
                return False

            return True

        except Exception:
            return False

    def extract_from_url(self, url: str) -> Tuple[str, Dict]:
        """
        Extract article content from URL using newspaper3k

        Args:
            url: Article URL

        Returns:
            Tuple of (extracted_text, metadata)
            metadata includes: title, author, publish_date, source, length

        Raises:
            ValueError: If URL is invalid
            Exception: If extraction fails
        """
        if not self.is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        try:
            logger_tools.debug(f"Extracting from URL: {url}")

            # Download and parse article
            article = Article(url, headers={"User-Agent": self.USER_AGENT})
            article.download()
            article.parse()

            # Extract text
            text = article.text.strip()

            if not text or len(text) < self.MIN_TEXT_LENGTH:
                raise ValueError(
                    f"Extracted text too short ({len(text)} chars). "
                    "URL might not be a news article."
                )

            # Prepare metadata
            metadata = {
                "source": "url",
                "url": url,
                "title": article.title or "Unknown",
                "author": article.authors[0] if article.authors else "Unknown",
                "publish_date": str(article.publish_date)
                if article.publish_date
                else "Unknown",
                "length": len(text),
                "word_count": len(text.split()),
            }

            logger_tools.info(
                f"✅ Extracted {len(text)} chars from: {article.title or url}"
            )
            return text, metadata

        except Exception as e:
            logger_tools.error(f"❌ URL extraction failed: {str(e)}")
            raise

    def extract_from_raw_text(self, text: str) -> Tuple[str, Dict]:
        """
        Process raw article text

        Args:
            text: Raw article text

        Returns:
            Tuple of (cleaned_text, metadata)

        Raises:
            ValueError: If text is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        text = text.strip()

        if len(text) < self.MIN_TEXT_LENGTH:
            raise ValueError(
                f"Text too short ({len(text)} chars). "
                f"Minimum {self.MIN_TEXT_LENGTH} characters required."
            )

        try:
            logger_tools.debug(f"Processing raw text ({len(text)} chars)")

            metadata = {
                "source": "raw_text",
                "url": None,
                "title": "User Input",
                "author": "Unknown",
                "publish_date": "Unknown",
                "length": len(text),
                "word_count": len(text.split()),
            }

            logger_tools.info(f"✅ Processed raw text ({len(text)} chars)")
            return text, metadata

        except Exception as e:
            logger_tools.error(f"❌ Raw text processing failed: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Text to clean

        Returns:
            Cleaned text

        Note:
            - Preserves negation words
            - Removes extra whitespace
            - Removes special characters
            - Normalizes spacing
        """
        try:
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove HTML entities
            text = re.sub(r"&[a-z]+;", "", text)

            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)

            # Keep alphanumeric, spaces, and basic punctuation
            # Allow: letters, numbers, spaces, hyphens, apostrophes, commas, periods
            text = re.sub(r"[^a-zA-Z0-9\s\-\'\",.\!?\(\)]", "", text)

            # Remove multiple spaces
            text = re.sub(r"\s+", " ", text)

            text = text.strip()

            return text

        except Exception as e:
            logger_tools.error(f"Text cleaning failed: {str(e)}")
            raise

    def extract(
        self, input_data: str, input_type: str = "auto"
    ) -> Tuple[str, Dict]:
        """
        Main extraction function - handles both URL and raw text

        Args:
            input_data: URL or raw text
            input_type: "url", "text", or "auto" (auto-detect)

        Returns:
            Tuple of (extracted_text, metadata)

        Raises:
            ValueError: If input is invalid
            Exception: If extraction fails
        """
        if not input_data or not isinstance(input_data, str):
            raise ValueError("Input must be a non-empty string")

        input_data = input_data.strip()

        # Auto-detect input type if needed
        if input_type == "auto":
            if self.is_valid_url(input_data):
                input_type = "url"
            else:
                input_type = "text"

        logger_tools.debug(f"Extracting as {input_type}: {input_data[:50]}...")

        try:
            if input_type == "url":
                text, metadata = self.extract_from_url(input_data)
            elif input_type == "text":
                text, metadata = self.extract_from_raw_text(input_data)
            else:
                raise ValueError(
                    f"Invalid input_type: {input_type}. "
                    "Must be 'url', 'text', or 'auto'"
                )

            # Clean the extracted text
            cleaned_text = self.clean_text(text)

            logger_tools.info(
                f"✅ Extraction complete: "
                f"{len(cleaned_text)} chars, "
                f"{metadata['word_count']} words"
            )

            return cleaned_text, metadata

        except Exception as e:
            logger_tools.error(f"❌ Extraction failed: {str(e)}")
            raise


# Global instance
extractor = ContentExtractor()
