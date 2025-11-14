"""
Open Source Web Search Integration for GB10
Replaces Tavily with DuckDuckGo or SearXNG

This module provides unified interface for open source web search engines,
allowing the Agentic RAG application to work without proprietary APIs.

Supported engines:
- DuckDuckGo: Free, no setup required
- SearXNG: Self-hosted, maximum privacy

Author: NVIDIA
Date: 2025-11-14
License: Apache 2.0
"""

import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OpenSourceSearch:
    """Unified interface for open source web search"""

    def __init__(self, engine: str = "duckduckgo", searxng_url: Optional[str] = None):
        """
        Initialize open source search

        Args:
            engine: "duckduckgo" or "searxng"
            searxng_url: URL of SearXNG instance (required if engine="searxng")
        """
        self.engine = engine.lower()
        self.searxng_url = searxng_url

        if self.engine == "duckduckgo":
            self._init_duckduckgo()
        elif self.engine == "searxng":
            self._init_searxng()
        else:
            raise ValueError(f"Unknown search engine: {engine}")

    def _init_duckduckgo(self):
        """Initialize DuckDuckGo search"""
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            logger.info("✅ DuckDuckGo search initialized (no API key required)")
        except ImportError:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Run: pip install duckduckgo-search"
            )

    def _init_searxng(self):
        """Initialize SearXNG search"""
        if not self.searxng_url:
            self.searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8888")

        logger.info(f"✅ SearXNG search initialized: {self.searxng_url}")

    def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Search the web

        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results with 'title', 'url', 'content'
        """
        if self.engine == "duckduckgo":
            return self._search_duckduckgo(query, max_results, **kwargs)
        elif self.engine == "searxng":
            return self._search_searxng(query, max_results, **kwargs)

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Search using DuckDuckGo"""
        try:
            results = []

            # Text search
            ddg_results = self.ddgs.text(
                query,
                max_results=max_results,
                **kwargs
            )

            for r in ddg_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "source": "duckduckgo"
                })

            logger.info(f"DuckDuckGo search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def _search_searxng(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Search using SearXNG"""
        import requests

        try:
            # SearXNG search API
            params = {
                "q": query,
                "format": "json",
                "categories": kwargs.get("categories", "general"),
                "engines": kwargs.get("engines", "google,duckduckgo,bing")
            }

            response = requests.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for r in data.get("results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "source": "searxng",
                    "engine": r.get("engine", "")
                })

            logger.info(f"SearXNG search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            logger.warning(f"Make sure SearXNG is running at: {self.searxng_url}")
            return []


# LangChain-compatible wrapper
class OpenSourceSearchTool:
    """LangChain-compatible tool for open source search"""

    def __init__(self, engine: str = "duckduckgo", searxng_url: Optional[str] = None):
        self.search = OpenSourceSearch(engine, searxng_url)

    def run(self, query: str, **kwargs) -> str:
        """Run search and return formatted results"""
        results = self.search.search(query, **kwargs)

        if not results:
            return "No results found."

        # Format results
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   URL: {r['url']}\n"
                f"   {r['content']}\n"
            )

        return "\n".join(formatted)


def get_search_tool(
    engine: Optional[str] = None,
    searxng_url: Optional[str] = None
) -> OpenSourceSearchTool:
    """
    Get configured search tool

    Args:
        engine: Search engine to use (from env if not specified)
        searxng_url: SearXNG URL (from env if not specified)

    Returns:
        Configured search tool
    """
    if engine is None:
        engine = os.getenv("SEARCH_ENGINE", "duckduckgo")

    if searxng_url is None:
        searxng_url = os.getenv("SEARXNG_URL")

    return OpenSourceSearchTool(engine, searxng_url)


# Backwards compatibility with Tavily interface
class TavilySearchResults:
    """
    Drop-in replacement for TavilySearchResults using open source search

    This provides backwards compatibility for code expecting Tavily API
    """

    def __init__(self, max_results: int = 5, search_engine: str = "duckduckgo"):
        """
        Initialize Tavily-compatible search

        Args:
            max_results: Maximum number of results to return
            search_engine: Which search engine to use
        """
        self.max_results = max_results
        self.search = OpenSourceSearch(engine=search_engine)
        logger.info(
            f"Using {search_engine} as Tavily replacement "
            f"(max_results={max_results})"
        )

    def invoke(self, input_dict: Dict) -> List[Dict]:
        """
        Invoke search (Tavily-compatible interface)

        Args:
            input_dict: Dictionary with 'query' key

        Returns:
            List of dicts with 'content', 'url', 'title' keys
        """
        query = input_dict.get("query", "")

        results = self.search.search(
            query=query,
            max_results=self.max_results
        )

        # Convert to Tavily-compatible format
        tavily_format = []
        for r in results:
            tavily_format.append({
                "content": r.get("content", ""),
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "source": r.get("source", "opensource")
            })

        return tavily_format


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("Open Source Search Test")
    print("="*60)

    # Test DuckDuckGo
    print("\n[1/2] Testing DuckDuckGo...")
    try:
        ddg_search = OpenSourceSearch(engine="duckduckgo")
        results = ddg_search.search("Python programming", max_results=3)

        print(f"✅ DuckDuckGo returned {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Title: {r['title'][:60]}...")
            print(f"    URL: {r['url']}")

    except Exception as e:
        print(f"❌ DuckDuckGo test failed: {e}")

    # Test Tavily compatibility mode
    print("\n[2/2] Testing Tavily Compatibility Mode...")
    try:
        tavily_compat = TavilySearchResults(max_results=2)
        results = tavily_compat.invoke({"query": "Machine learning"})

        print(f"✅ Tavily-compatible mode returned {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Title: {r['title'][:60]}...")
            print(f"    Content: {r['content'][:100]}...")

    except Exception as e:
        print(f"❌ Tavily compatibility test failed: {e}")

    print("\n" + "="*60)
    print("✅ Open source search is ready!")
    print("="*60)
