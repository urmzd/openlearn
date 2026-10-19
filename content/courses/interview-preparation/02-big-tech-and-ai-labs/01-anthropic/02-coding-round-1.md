# The First Coding Round: Concurrent Systems

The first live coding round is 60 minutes in a shared editor (CoderPad or similar) over video, and Python is strongly preferred. Where the [online assessment](01-online-assessment.md) asked whether your code looks shippable, this round asks whether you can reason about concurrency while you write it. The problems are real-world engineering problems rather than puzzles, and the interviewer is as interested in your explanation of the tradeoffs as in the code.

This section works through the flagship problem, a concurrent web crawler, in full: the requirements, an annotated asyncio implementation, and the follow-up questions interviewers ask about it. It then covers two other reported problems, parallel word segmentation and a stack-sampling profiler, and closes with the concurrency background you need to have at your fingertips: the GIL, the asyncio event loop, and the difference between coroutines and threads.

> **Learning objectives**
> - Build a breadth-first, depth-limited, rate-limited asynchronous web crawler with deduplication and a site map.
> - Defend the choice of asyncio over threads for I/O-bound work, and the use of a semaphore for concurrency limits.
> - Handle the crawler's edge cases: redirect loops, relative and protocol-relative URLs, hanging pages, and memory pressure at scale.
> - Solve word segmentation with dynamic programming over suffixes.
> - Explain what the GIL does and does not protect, and when to reach for asyncio, threads, or multiprocessing.

## The concurrent web crawler

The crawler is the problem you are most likely to see. You are asked to build a web crawler that performs a breadth-first traversal from a seed URL; respects a configurable depth limit; extracts and follows links from HTML content; builds a site map from each URL to the list of URLs it links to; deduplicates visited URLs; limits itself to at most N concurrent requests; and, as a bonus, honors `robots.txt`.

### Designing the crawler

Each requirement maps to one piece of the design. Breadth-first traversal with a depth limit means processing the frontier one level at a time and tagging each URL with its depth. Concurrency within a level comes from `asyncio.gather` over the level's URLs. The concurrency cap is an `asyncio.Semaphore` held around each fetch. Deduplication is a `visited` set that a URL joins *when it is discovered*, not when it is fetched, so the same link found on two pages in one level is enqueued only once. And link extraction uses the standard library's `HTMLParser` plus a normalizer that resolves relative URLs, drops non-HTTP schemes, and strips fragments.

### The implementation

Here is the complete reference solution.

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

A few details are worth pointing out as you walk the interviewer through it. `crawl` copies the frontier into `current_batch` and clears `queue`, so each pass of the `while` loop is exactly one BFS level, and URLs deeper than `max_depth` are skipped before any fetch is scheduled. `_fetch` returns `None` on any client error, timeout, non-200 status, or non-HTML content type, so one failed request never kills the whole crawl; `_process_url` turns that `None` into an empty list of new URLs. And because asyncio runs on one thread and the check-and-add on `visited` contains no `await`, it cannot be interleaved by another coroutine, so no lock is needed around it.

### The follow-up questions

Interviewers reliably probe the same six points, and you should have an answer ready for each.

*Why asyncio over threading?* Web crawling is I/O-bound. asyncio avoids per-thread overhead and GIL contention, and a single event loop efficiently multiplexes thousands of connections.

*Why a semaphore?* `asyncio.Semaphore(N)` caps the number of concurrent requests. Be ready to discuss alternatives: a token bucket if the limit is requests per second rather than requests in flight, and per-domain limits so you are polite to each host.

*What about redirect loops?* The client's `max_redirects` bounds them, and you can also add redirected URLs to `visited`.

*Relative versus absolute URLs?* `urljoin` handles the common case, but discuss the edge cases: protocol-relative links (`//example.com`), query parameters (the normalizer above drops them, which may merge distinct pages), and fragments (stripped, correctly, since they name a position within one page).

*What about hanging pages?* The `timeout` prevents indefinite hangs. Discuss what happens when a page streams slowly: a total timeout cuts it off, which you may or may not want.

*What about memory at scale?* For large crawls, discuss bounded queues, disk-backed or probabilistic visited sets (Bloom filters), and streaming the site map out instead of holding it all in memory.

## Parallel word segmentation

The second reported problem gives you a dictionary and a string with no spaces and asks for all valid segmentations of the string into dictionary words, with a follow-up about parallelizing across starting positions.

The core is dynamic programming over suffixes. Let `dp[i]` be the list of segmentations of `text[i:]`, with `dp[n] = [[]]` (the empty suffix has one segmentation, the empty one). Working right to left, `dp[i]` collects `[word] + rest` for every dictionary word `text[i:j]` and every segmentation `rest` of `dp[j]`. Capping word length at `max_word_len` bounds the inner loop. A suffix with no segmentations is marked `None`, so later positions skip it immediately.

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

The function is declared `async` but contains no `await`; the DP itself is sequential and CPU-bound. If the interviewer asks you to parallelize, say so honestly: under the GIL, coroutines and threads give no speedup for CPU work, so real parallelism across starting positions means `multiprocessing` or a process pool, and the dependency of `dp[i]` on later entries limits how much can run independently. Also note that the number of segmentations can be exponential in the string length, so the output size, not the DP, may dominate.

## The stack-sampling profiler

The third reported problem converts the output of a stack-sampling profiler into trace events. Given periodic snapshots of call stacks, you produce enter and exit events by diffing consecutive samples. The detailed version, with a full solution, is in [the second coding round](04-coding-round-2.md).

## The concurrency background you need

All three problems assume fluency with Python's concurrency model. Three topics come up again and again.

### The GIL

CPython's Global Interpreter Lock means only one thread executes Python bytecode at a time. For I/O-bound work, use `asyncio` or `threading`; the GIL is released during I/O operations, so threads still overlap their waiting. For CPU-bound work, use `multiprocessing` to bypass the GIL entirely. The GIL makes individual operations on built-in types, such as a list append or a dict access, atomic, but compound operations like check-then-set still need locks.

### The asyncio event loop

The core asyncio pattern is to create tasks and gather their results:

```python
# Core pattern: create tasks, gather results
async def main():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            handle_error(result)
```

`await` suspends the coroutine and yields control to the event loop. `asyncio.gather()` runs coroutines *concurrently*, not in parallel, since everything is on a single thread; `return_exceptions=True` keeps one failure from cancelling the rest. `asyncio.Semaphore` provides concurrency control and `asyncio.Queue` provides producer-consumer patterns.

> **Warning:** Never block the event loop with synchronous I/O or long CPU work. A single blocking call stalls every coroutine. Use `loop.run_in_executor()` to push blocking work to a thread or process pool.

### Coroutines versus threads

The tradeoffs between the two models are summarized below.

| Aspect | Coroutines (asyncio) | Threads |
|--------|---------------------|---------|
| Concurrency model | Cooperative (explicit yield) | Preemptive (OS-scheduled) |
| Overhead | Very low (~KB per coroutine) | Higher (~MB per thread) |
| Scaling | 10K+ concurrent tasks | Hundreds at most |
| Best for | I/O-bound, network | I/O-bound, legacy sync libs |
| Debugging | Easier (deterministic switching) | Harder (race conditions) |
| GIL impact | N/A (single thread) | Limits CPU parallelism |

## Preparing for the round

Know asyncio cold: you should be able to write the crawler from scratch without looking anything up, including the `async with`, `async for`, `asyncio.gather`, and semaphore patterns. Think out loud, because the interviewer wants to hear your reasoning about concurrency tradeoffs. Start simple and extend; a synchronous version that works is a better foundation than a concurrent one that does not. Handle errors gracefully so one failed request cannot kill the whole job. And discuss scale even if you are not asked: say what you would change for a million URLs versus a hundred.

## Where this leads

The deeper treatment of locks, asyncio primitives, and distributed consistency is in [Concurrency patterns](06-concurrency-patterns.md). The profiler problem is solved in full in [the second coding round](04-coding-round-2.md), and the crawler's ideas of bounded concurrency and backpressure return in the [system design round](03-system-design.md) and in [Streaming and SSE](08-streaming-sse.md).

## Key terms

- **BFS level**: the set of URLs at the same distance from the seed, processed together in one pass.
- **Semaphore**: a counter that limits how many coroutines may hold it at once; here, the number of concurrent requests.
- **Event loop**: asyncio's single-threaded scheduler, which runs coroutines until they `await` and then switches.
- **GIL**: CPython's Global Interpreter Lock, which allows only one thread to run Python bytecode at a time.
- **Cooperative concurrency**: a model in which tasks switch only at explicit yield points, as opposed to preemptive OS scheduling.

## Summary

- The crawler is BFS by level, `asyncio.gather` within a level, a semaphore around each fetch, and a `visited` set updated at discovery time.
- Be ready for the six follow-ups: asyncio versus threads, rate-limiting alternatives, redirects, URL normalization, timeouts, and memory at scale.
- Word segmentation is DP over suffixes; parallelizing CPU-bound work in Python needs processes, not coroutines.
- The GIL serializes bytecode but is released for I/O; compound operations still need locks.
- Never block the event loop; push blocking work to an executor.

## Review questions

1. Why is no lock needed around the `visited` set in the asyncio crawler?
   <details><summary>Answer</summary>asyncio runs every coroutine on one thread and switches only at <code>await</code>. The check <code>if link not in self.visited</code> and the following <code>add</code> contain no <code>await</code>, so no other coroutine can run between them.</details>
2. Why are URLs added to `visited` when they are discovered rather than when they are fetched?
   <details><summary>Answer</summary>If two pages in the same level link to the same URL, marking it at discovery means only the first discovery enqueues it. Marking at fetch time would let both copies into the next level and fetch the page twice.</details>
3. The requirement changes from "at most 10 requests in flight" to "at most 10 requests per second." What changes?
   <details><summary>Answer</summary>A semaphore bounds in-flight requests, not rate. A rate limit needs a token bucket (or similar) that refills at 10 tokens per second, with each request taking a token; for politeness you would likely keep one bucket per domain.</details>
4. An interviewer asks you to speed up `segment` by making it concurrent with asyncio. What do you say?
   <details><summary>Answer</summary>The work is CPU-bound and has no I/O to overlap, so coroutines (and, under the GIL, threads) give no speedup. Real parallelism would need multiple processes, and the right-to-left dependency between DP entries limits how much can run independently. The output can also be exponentially large, which may dominate the cost.</details>
5. Is `my_dict[key] = my_dict.get(key, 0) + 1` safe across threads under the GIL?
   <details><summary>Answer</summary>No. The read and the write are separate operations, and another thread can run between them, losing an increment. The GIL makes single operations on built-ins atomic, not compound read-modify-write sequences; you need a lock.</details>
