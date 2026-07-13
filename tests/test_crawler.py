from unittest.mock import patch
import pytest
from src.personal_ai_engine.domain import Document, DocumentMetadata
from src.offline.crawlers.crawl4ai import Crawl4AICrawler


@pytest.mark.asyncio
async def test_crawler_deduplication():
    meta = DocumentMetadata(
        id="1", url="https://notion.so/1", title="Test Page", properties={}
    )
    doc = Document(
        id="1",
        metadata=meta,
        content="Test",
        child_urls=["https://example.com", "https://example.com/2"],
    )

    class MockResult:
        success = True
        markdown = "Crawled content"
        links = {"internal": [{"href": "https://example.com/2"}], "external": []}
        metadata = {"title": "Crawled"}

    class MockAsyncCrawler:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def arun(self, url):
            return MockResult()

    with patch(
        "src.offline.crawlers.crawl4ai.AsyncWebCrawler", return_value=MockAsyncCrawler()
    ):
        crawler = Crawl4AICrawler(max_concurrent_requests=2)
        results = await crawler._Crawl4AICrawler__crawl_batch([doc])

        assert len(results) == 2

        # Deduplicate using set logic mimicking the pipeline step
        augmented_pages = [doc]
        augmented_pages.extend(results)
        augmented_pages = list(set(augmented_pages))

        assert len(augmented_pages) == 3  # original doc + 2 newly crawled pages
