import asyncio
import os
import psutil
import urllib.parse
import ipaddress
import socket
from crawl4ai import AsyncWebCrawler, CacheMode
from loguru import logger

from src.personal_ai_engine import utils
from src.personal_ai_engine.domain import Document, DocumentMetadata
from src.shared.config import settings


def is_safe_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        if not host:
            return False

        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except socket.gaierror:
            pass

        allowed = settings.CRAWLER_ALLOWED_DOMAINS
        blocked = settings.CRAWLER_BLOCKED_DOMAINS

        if allowed and not any(host.endswith(d) for d in allowed):
            return False
        if blocked and any(host.endswith(d) for d in blocked):
            return False
        return True
    except Exception:
        return False


class Crawl4AICrawler:
    def __init__(
        self, max_concurrent_requests: int = settings.CRAWLER_CONCURRENCY
    ) -> None:
        self.max_concurrent_requests = max_concurrent_requests

    def __call__(self, pages: list[Document]) -> list[Document]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.__crawl_batch(pages))
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.__crawl_batch(pages)).result()

    async def __crawl_batch(self, pages: list[Document]) -> list[Document]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        all_results = []

        # Track seen URLs to avoid duplicate crawling
        seen_urls = set()

        async with AsyncWebCrawler(cache_mode=CacheMode.BYPASS) as crawler:
            for page in pages:
                tasks = []
                for url in page.child_urls:
                    if url not in seen_urls and is_safe_url(url):
                        seen_urls.add(url)
                        tasks.append(self.__crawl_url(crawler, page, url, semaphore))
                if tasks:
                    results = await asyncio.gather(*tasks)
                    all_results.extend(results)

        successful_results = [result for result in all_results if result is not None]
        return successful_results

    async def __crawl_url(
        self,
        crawler: AsyncWebCrawler,
        page: Document,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> Document | None:
        async with semaphore:
            retries = 3
            for attempt in range(retries):
                try:
                    result = await crawler.arun(
                        url=url,
                        user_agent=settings.CRAWLER_USER_AGENT,
                        word_count_threshold=10,
                        bypass_cache=True,
                    )
                    await asyncio.sleep(0.5)

                    if not result or not result.success:
                        if attempt == retries - 1:
                            logger.warning(f"Failed to crawl {url} permanently")
                            return None
                        await asyncio.sleep(2**attempt)
                        continue

                    if (
                        result.markdown is None
                        or len(result.markdown) > settings.CRAWLER_MAX_RESPONSE_SIZE
                    ):
                        logger.warning(f"Failed to crawl {url} or response too large")
                        return None

                    child_links = []
                    for link in result.links.get("internal", []) + result.links.get(
                        "external", []
                    ):
                        if is_safe_url(link.get("href", "")):
                            child_links.append(link["href"])

                    title = result.metadata.pop("title", "") if result.metadata else ""
                    document_id = utils.generate_random_hex(length=32)

                    return Document(
                        id=document_id,
                        metadata=DocumentMetadata(
                            id=document_id,
                            url=url,
                            title=title,
                            properties=result.metadata or {},
                        ),
                        parent_metadata=page.metadata,
                        content=str(result.markdown),
                        child_urls=list(set(child_links)),
                    )
                except Exception as e:
                    logger.warning(f"Transient error crawling {url}: {e}")
                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(2**attempt)
            return None
