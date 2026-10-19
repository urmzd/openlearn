# Round 2: Coding Round 1 (Live)

## Format

- **Duration**: 60 minutes
- **Setting**: Live video with shared editor (CoderPad or similar)
- **Focus**: Concurrent/async systems, real-world engineering problems
- **Language**: Python strongly preferred

## Reported Problems

### 1. Concurrent Web Crawler

The flagship problem for this round. Build a web crawler that:

- Performs BFS traversal starting from a seed URL
- Respects a configurable depth limit
- Extracts and follows links from HTML content
- Builds a site map (URL -> list of linked URLs)
- Deduplicates visited URLs
- Implements rate limiting (max N concurrent requests)
- Handles robots.txt (bonus)

#### Core Implementation

```python
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from collections import defaultdict

class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, value in attrs:
                if attr == "href" and value:
                    self.links.append(value)

class WebCrawler:
    def __init__(self, max_depth: int, max_concurrent: int, timeout: float = 10.0):
        self.max_depth = max_depth
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.visited: set[str] = set()
        self.site_map: dict[str, list[str]] = defaultdict(list)

    def _normalize_url(self, base: str, url: str) -> Optional[str]:
        """Resolve relative URLs and normalize."""
        resolved = urljoin(base, url)
        parsed = urlparse(resolved)
        if parsed.scheme not in ("http", "https"):
            return None
        # Strip fragment, normalize
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        parser = LinkExtractor()
        parser.feed(html)
        links = []
        for link in parser.links:
            normalized = self._normalize_url(base_url, link)
            if normalized:
                links.append(normalized)
        return links

    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a URL with rate limiting and timeout."""
        async with self.semaphore:
            try:
                async with session.get(url, timeout=self.timeout,
                                       allow_redirects=True,
                                       max_redirects=5) as response:
                    if response.status == 200 and "text/html" in response.headers.get("content-type", ""):
                        return await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            return None

    async def crawl(self, seed_url: str) -> dict[str, list[str]]:
        """BFS crawl starting from seed_url."""
        async with aiohttp.ClientSession() as session:
            queue = [(seed_url, 0)]  # (url, depth)
            self.visited.add(seed_url)

            while queue:
                # Process current level in parallel
                current_batch = queue[:]
                queue.clear()

                tasks = []
                for url, depth in current_batch:
                    if depth > self.max_depth:
                        continue
                    tasks.append(self._process_url(session, url, depth))

                results = await asyncio.gather(*tasks)
                for new_urls in results:
                    queue.extend(new_urls)

        return dict(self.site_map)

    async def _process_url(self, session, url: str, depth: int) -> list[tuple[str, int]]:
        """Fetch and extract links from a single URL."""
        html = await self._fetch(session, url)
        if html is None:
            return []

        links = self._extract_links(html, url)
        self.site_map[url] = links

        new_urls = []
        for link in links:
            if link not in self.visited:
                self.visited.add(link)
                new_urls.append((link, depth + 1))
        return new_urls
```

#### Key Discussion Points

The interviewer will probe on:

1. **Why asyncio over threading?** -- Web crawling is I/O-bound. asyncio avoids thread overhead and GIL contention. A single event loop efficiently multiplexes thousands of connections.

2. **Semaphore for rate limiting** -- `asyncio.Semaphore(N)` caps concurrent requests. Discuss alternatives: token bucket for requests-per-second, per-domain rate limiting.

3. **Redirect loops** -- `max_redirects` on the client session, plus tracking redirected URLs in `visited`.

4. **Relative vs. absolute URLs** -- `urljoin` handles this, but discuss edge cases: protocol-relative (`//example.com`), query parameters, fragments.

5. **Hanging pages** -- The `timeout` parameter prevents indefinite hangs. Discuss what happens when a page streams slowly.

6. **Memory pressure** -- For large crawls, discuss bounded queues, disk-backed visited sets (bloom filters), or streaming output.

### 2. Parallel Word Segmentation

Given a dictionary and a string with no spaces, find all valid word segmentations. Parallelize across multiple starting positions.

```python
async def segment(text: str, dictionary: set[str], max_word_len: int = 20) -> list[list[str]]:
    """Find all valid segmentations of text into dictionary words."""
    n = len(text)
    # dp[i] = list of segmentations for text[i:]
    dp: list[Optional[list[list[str]]]] = [None] * (n + 1)
    dp[n] = [[]]

    for i in range(n - 1, -1, -1):
        segmentations = []
        for j in range(i + 1, min(i + max_word_len + 1, n + 1)):
            word = text[i:j]
            if word in dictionary and dp[j] is not None:
                for rest in dp[j]:
                    segmentations.append([word] + rest)
        dp[i] = segmentations if segmentations else None

    return dp[0] or []
```

### 3. Stack Sampling Profiler

Convert stack sampling profiler output into trace events. Given periodic snapshots of call stacks, produce enter/exit events by diffing consecutive samples.

See [Coding Round 2](04-coding-round-2.md) for the detailed version of this problem.

## Key Concepts to Know

### Python GIL (Global Interpreter Lock)

- Only one thread executes Python bytecode at a time
- **I/O-bound work**: Use `asyncio` or `threading` -- GIL is released during I/O operations
- **CPU-bound work**: Use `multiprocessing` to bypass the GIL entirely
- The GIL makes thread-safe operations on built-in types (list append, dict access) atomic, but compound operations still need locks

### asyncio Event Loop

```python
# Core pattern: create tasks, gather results
async def main():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            handle_error(result)
```

- `await` suspends the coroutine and yields control to the event loop
- `asyncio.gather()` runs coroutines concurrently (not in parallel -- single thread)
- `asyncio.Semaphore` for concurrency control
- `asyncio.Queue` for producer-consumer patterns
- Never block the event loop with synchronous I/O -- use `loop.run_in_executor()` if needed

### Coroutines vs. Threads

| Aspect | Coroutines (asyncio) | Threads |
|--------|---------------------|---------|
| Concurrency model | Cooperative (explicit yield) | Preemptive (OS-scheduled) |
| Overhead | Very low (~KB per coroutine) | Higher (~MB per thread) |
| Scaling | 10K+ concurrent tasks | Hundreds at most |
| Best for | I/O-bound, network | I/O-bound, legacy sync libs |
| Debugging | Easier (deterministic switching) | Harder (race conditions) |
| GIL impact | N/A (single thread) | Limits CPU parallelism |

## Preparation Tips

1. **Know asyncio cold** -- Write a crawler from scratch without looking anything up. Practice the `async with`, `async for`, `asyncio.gather`, and semaphore patterns.
2. **Think out loud** -- The interviewer wants to hear your reasoning about concurrency trade-offs.
3. **Start simple, extend** -- Get a synchronous version working first, then add concurrency.
4. **Handle errors gracefully** -- Don't let one failed request kill the whole crawl.
5. **Discuss scale** -- Even if not asked, mention what you'd change for 1M URLs vs. 100.
