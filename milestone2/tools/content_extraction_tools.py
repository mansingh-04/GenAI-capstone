"""
Content Extraction Tools
Exposed functions for content extraction via API and agent
"""

from typing import Dict, Tuple
from logger import logger_tools
from tools.content_extractor import extractor


def extract_content(
    input_data: str, input_type: str = "auto"
) -> Dict:
    """
    Extract and clean article content from URL or raw text

    Main function used by API and agent for content extraction.

    Args:
        input_data: URL or raw article text
        input_type: "url", "text", or "auto" (default: auto-detect)

    Returns:
        Dictionary containing:
        - content: Cleaned article text
        - metadata: Dict with title, author, source, length, etc.
        - success: Boolean indicating success
        - error: Error message if failed

    Example:
        >>> result = extract_content("https://example.com/article")
        >>> print(result['content'][:100])
        >>> print(result['metadata']['title'])

        >>> result = extract_content("The article text here...")
        >>> print(result['content'])
    """
    try:
        logger_tools.debug(f"Extracting content (type: {input_type})")

        # Extract content
        content, metadata = extractor.extract(input_data, input_type)

        result = {
            "success": True,
            "content": content,
            "metadata": metadata,
            "error": None,
        }

        logger_tools.info(
            f"✅ Content extracted: {len(content)} chars, "
            f"{metadata['word_count']} words from {metadata['source']}"
        )
        return result

    except ValueError as e:
        logger_tools.warning(f"⚠️  Validation error: {str(e)}")
        return {
            "success": False,
            "content": None,
            "metadata": None,
            "error": f"Invalid input: {str(e)}",
        }

    except Exception as e:
        logger_tools.error(f"❌ Content extraction failed: {str(e)}")
        return {
            "success": False,
            "content": None,
            "metadata": None,
            "error": f"Extraction failed: {str(e)}",
        }


def clean_article_text(text: str) -> str:
    """
    Clean and normalize article text

    Args:
        text: Raw article text

    Returns:
        Cleaned text

    Example:
        >>> dirty = "Check  out  this!!!  text   with   spaces"
        >>> clean = clean_article_text(dirty)
        >>> print(clean)  # "Check out this text with spaces"
    """
    try:
        logger_tools.debug(f"Cleaning text ({len(text)} chars)")
        cleaned = extractor.clean_text(text)
        logger_tools.info(f"✅ Text cleaned: {len(text)} → {len(cleaned)} chars")
        return cleaned

    except Exception as e:
        logger_tools.error(f"❌ Text cleaning failed: {str(e)}")
        raise


def validate_url(url: str) -> Dict:
    """
    Validate if a URL is a valid news article URL

    Args:
        url: URL to validate

    Returns:
        Dictionary with:
        - is_valid: Boolean
        - message: Description

    Example:
        >>> result = validate_url("https://example.com/article")
        >>> print(result['is_valid'])  # True or False
    """
    try:
        is_valid = extractor.is_valid_url(url)
        logger_tools.debug(f"URL validation: {url} → {is_valid}")
        return {
            "url": url,
            "is_valid": is_valid,
            "message": "Valid URL" if is_valid else "Invalid URL format",
        }

    except Exception as e:
        logger_tools.error(f"URL validation error: {str(e)}")
        return {
            "url": url,
            "is_valid": False,
            "message": f"Validation error: {str(e)}",
        }


def batch_extract_content(inputs: list) -> list:
    """
    Extract content from multiple inputs

    Args:
        inputs: List of dicts with 'data' and optional 'type' keys
                Example: [
                    {'data': 'https://...', 'type': 'url'},
                    {'data': 'Article text...', 'type': 'text'},
                    {'data': 'Mixed...'}  # auto-detect
                ]

    Returns:
        List of extraction results

    Example:
        >>> inputs = [
        ...     {'data': 'https://example.com/1'},
        ...     {'data': 'Article text here'}
        ... ]
        >>> results = batch_extract_content(inputs)
        >>> print(len(results))  # 2
    """
    try:
        logger_tools.info(f"Batch extracting from {len(inputs)} inputs")
        results = []

        for i, item in enumerate(inputs):
            try:
                data = item.get("data")
                input_type = item.get("type", "auto")

                result = extract_content(data, input_type)
                results.append(result)

            except Exception as e:
                logger_tools.warning(f"Batch item {i} failed: {str(e)}")
                results.append(
                    {
                        "success": False,
                        "content": None,
                        "metadata": None,
                        "error": str(e),
                    }
                )

        logger_tools.info(
            f"✅ Batch extraction complete: "
            f"{sum(1 for r in results if r['success'])}/{len(results)} successful"
        )
        return results

    except Exception as e:
        logger_tools.error(f"❌ Batch extraction failed: {str(e)}")
        return []
