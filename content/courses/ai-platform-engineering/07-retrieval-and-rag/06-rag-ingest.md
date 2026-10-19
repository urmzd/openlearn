# RAG Ingest: Crawler, Chunker, Embedder, and Store

The agent you built in [The agent loop](../13-agent-sdk/03-agent-loop.md) knows only what its model memorized. The goal of this part of the book is an agent that answers questions about the course and its own docs, with citations. That means retrieval: find the passages that answer the question and put them in the context. Retrieval, covered in the next section, searches an index. This section builds that index.

Ingest looks like plumbing, but each step has a way to quietly ruin the answers. Chunks longer than the embedding model's context are truncated by the engine and lose their ends without any error. A crawler that follows any link will fetch `http://169.254.169.254/latest/meta-data/` (the cloud credentials endpoint) simply because a document linked it. A nightly re-ingest that re-embeds everything costs hours. And vectors stored unnormalized make cosine similarity wrong. By the end of this section you will have a pipeline that avoids all four: sources that produce clean text, a chunker that packs to a true token budget, an embedder that cannot misalign rows, an index that skips unchanged documents, and a crawler that a hostile document cannot aim at your network.

This section builds on [Types, an OpenAI-compatible provider, and a retry wrapper](../13-agent-sdk/01-types-and-provider.md), whose `Retryable` and `RetryPolicy` the embedder reuses. It helps to have read [Tools, a registry, and JSON Schema argument validation](../13-agent-sdk/02-tools-and-schemas.md), the engine's tokenizer in [Fast BPE in Rust](../../build-an-llm/p01-tokenizers/05-rust-fast-bpe.md) from *Build Your Own LLM System*, and the crawler warm-up in [practice Go 01](../../../../practice/build/cloud/go/). The full solution lives in [`source.go`](../../../../course/solutions/go/agent/rag/source/source.go) (`RawDoc`, `Doc`, `ToDoc`, `DocID`, `DirSource`), [`crawler.go`](../../../../course/solutions/go/agent/rag/source/crawler.go) (`Crawler`, `PublicAddr`), [`chunk.go`](../../../../course/solutions/go/agent/rag/chunk/chunk.go) (`TokenChunker`, `Bytes`, `HTTPTokenizer`, `Fingerprint`), [`embed.go`](../../../../course/solutions/go/agent/rag/embed/embed.go) (`HTTPEmbedder`), and [`store.go`](../../../../course/solutions/go/agent/rag/store/store.go) (`Open`, `Index.Upsert`, `Index.Save`, `Ingest`). The on-disk format is fixed by the index directory of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md) (`meta.json`, `docs.jsonl`, `vectors.f32`), and the engine calls by `/v1/tokenize` and `/v1/embeddings` in [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml).

> **Learning objectives**
> - Pack a document into chunks under a token budget counted by the embedding model's own tokenizer, splitting at the largest natural unit that fits and never inside a UTF-8 character.
> - Make ingest idempotent by fingerprint, so an unchanged document embeds and writes nothing and a changed one re-embeds only its new chunks.
> - Store normalized vectors in the contract's index format with atomic writes.
> - Defend a crawler against SSRF with an exact host allowlist, a dial-time address check, and redirect re-checks.
> - Crawl politely: robots.txt per RFC 9309 and a per-host rate on a clock that can be faked.

## From documents to text

Everything starts with a **source**, which returns raw documents. `DirSource` reads files under a directory (`.md` and `.txt` by default, in path order); the **crawler** fetches web pages. `ToDoc` then turns a raw document into text. HTML loses the content of `<script>`, `<style>`, and `<head>` and then every tag; block tags (`<p>`, `<li>`, `<h1>`, ...) become paragraph breaks; entities are decoded after tags are removed (so an encoded `&lt;p&gt;` stays text); runs of spaces collapse; and paragraphs end up separated by exactly one blank line. The doc id is the URL without its fragment, with scheme and host lowercased, or the file path for a file.

The crawler goes breadth first from its seeds. It follows `<a href>` links resolved against the page's final URL (after redirects), drops `#fragments` so that `/a` and `/a#top` are one page, and visits each URL once, up to `MaxPages` and `MaxDepth`. Because it touches other people's servers, it obeys two politeness rules.

The first is **robots.txt** (RFC 9309), fetched once per host. The group naming our user agent applies, or else the `*` group. Within the group, the **longest** matching rule wins, and `Allow` wins a tie. In patterns, `*` matches any run of characters and a trailing `$` anchors the end. A 4xx robots.txt means the site has no restrictions; a 5xx or no answer at all means the site cannot say, so nothing is fetched.

The second is a **rate**: requests to one host, robots.txt included, are at least $1/\text{RPS}$ apart. Each host has its own next free slot, and the crawler sleeps only until that host's slot, so two hosts never wait for each other. Time comes from a `Clock`, which tests replace with a fake that advances only on `Sleep`.

## Chunking to a token budget

Clean text is still too long to embed whole, so the next job is to cut it into chunks. The notation is small:

| Symbol | Meaning |
|---|---|
| $B$ | the budget, `MaxTokens` |
| $T(x)$ | the token count of text $x$ by the embedding model's tokenizer |
| $p_1, \dots, p_k$ | the pieces of a document, each with $T(p_i) \le B$ |
| $s_i$ | the separator before $p_i$: `"\n\n"` between paragraphs, `" "` inside one |

The pieces are the document's paragraphs, except that a paragraph with $T(\text{paragraph}) > B$ is replaced by its sentences, a sentence that is still too long by its words, and a word that is still too long by its longest prefixes that fit, cut at a character boundary. Packing is then greedy. Keep a current chunk $c$; for each piece, if $T(c \,\Vert\, s_i \,\Vert\, p_i) \le B$, extend $c$; otherwise emit $c$ and start a new chunk with $p_i$.

The detail that matters most is that the count is taken on the **joined candidate**, never as $T(c) + T(p_i)$. Token counts do not add up across a join. A BPE tokenizer can merge characters across the join, so the sum overcounts and wastes context. A tokenizer that spends tokens on separators, or adds a marker per text, makes the sum undercount, which lets chunks overflow. Only the real tokenizer on the real text gives the right answer. `HTTPTokenizer` asks the engine (`/v1/tokenize`, and the count is the length of `ids`); `Bytes` is the tracer's byte-level tokenizer. The budget is inclusive.

Each chunk gets an id `<doc_id>#<k>`, where $k$ is its position in the document counting from 0, and a fingerprint, the hex SHA-256 of its text.

## Embedding and the index

`HTTPEmbedder` posts batches (64 inputs by default; the contract allows 256) to `/v1/embeddings` and places each vector by its `index` field, because a server may answer in any order. An answer with the wrong count, a repeated or out-of-range index, or vectors of different lengths is an error, never a misaligned index. Overload (503, 429) and broken connections are retried with the provider's policy.

The vectors land in the files `rag-index.md` fixes: `docs.jsonl` (one chunk per line; line $i$ is chunk $i$), `vectors.f32` ($n \times d$ little-endian float32, no header), and `meta.json` (`n_chunks`, `dim`, the models, the chunker and BM25 settings). Every row is L2-normalized when stored, so retrieval's cosine similarity is a plain dot product. A fourth file, `bm25.idx`, is written by retrieval from the same chunks.

`Ingest` ties it together with fingerprint dedup. It chunks each document and compares the chunk fingerprints with the ones stored for that doc id. If they are all equal, the document is **unchanged**: nothing is embedded and the index is not marked dirty. Otherwise the chunks whose fingerprint the document already had keep their stored vectors, only the new ones are embedded, and the document's chunks are replaced in place. `Save` writes nothing when nothing changed; otherwise it writes each file to a temporary name, fsyncs, and renames, so a reader never sees half a file.

## Server-side request forgery

The crawler deserves its own threat model. It is a program that makes HTTP requests to addresses chosen by its input. If the input is a document an attacker wrote, the attacker chooses where your server connects: an admin port on localhost, a database on the private network, the cloud metadata service at `169.254.169.254`. Three checks defend against this, in this order.

1. **Host allowlist**, on the URL's host name before any request is made. Match exact names only: a prefix match lets `docs.example.evil.example` in.
2. **Address check at dial time.** The crawler resolves the name itself, refuses if **any** address is not public, and dials the address it checked. Checking a name and then letting the HTTP library resolve it again opens a race: with DNS rebinding, the second answer is `127.0.0.1`. "Not public" means loopback, private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), link-local (`169.254/16`, `fe80::/10`), unspecified, multicast, carrier-grade NAT (`100.64/10`), NAT64 (`64:ff9b::/96`, which embeds an IPv4 address), and `0/8`. An IPv4-mapped IPv6 address (`::ffff:10.0.0.1`) is checked as the IPv4 address it really is.
3. **Redirects** are new requests. The allowlist is checked again for each target (at most 5), and each target's connection goes through the same dial check.

## Working it by hand

Take the document `docs/kv.md`, the byte tokenizer ($T(x)$ is the number of bytes of $x$), and a budget of $B = 60$:

```
Paged attention stores KV in blocks.                                  (36 bytes)

Blocks are fixed size. A block table maps token positions to blocks.  (68 bytes)

Eviction frees whole blocks.                                          (28 bytes)
```

Paragraph 2 alone is $68 > 60$, so it becomes its two sentences, of 22 and 45 bytes. The table traces the pieces through the packer:

| Piece | Separator | Candidate | $T$ | Fits? | Chunk so far |
|---|---|---|---|---|---|
| P1 (36) | | P1 | 36 | yes | P1 |
| S1 (22) | `\n\n` | P1 + `\n\n` + S1 | 60 | yes (inclusive) | P1, S1 |
| S2 (45) | `" "` | ... + `" "` + S2 | 106 | no: emit chunk 0 (60) | S2 |
| P3 (28) | `\n\n` | S2 + `\n\n` + P3 | 75 | no: emit chunk 1 (45) | P3 |
| end | | | | emit chunk 2 (28) | |

The result is three chunks: `docs/kv.md#0` (60 tokens, paragraph 1 and the first sentence), `#1` (45), and `#2` (28), each with the SHA-256 of its text as fingerprint. Notice that the second candidate is exactly $60 = B$ only because the join `\n\n` is counted: $36 + 2 + 22$.

Now re-ingest. The unchanged file gives the same three fingerprints, so nothing is embedded and `Save` writes nothing. Change the last paragraph to "Eviction frees whole blocks at once." and only chunk 2's fingerprint changes: one text is embedded.

## Designing the packages

With the behavior pinned down, the API falls out as four small packages:

```go
package source // import "tinyllm/agent/rag/source"
type RawDoc struct { URI, Source, ContentType string; Body []byte }
type Doc struct { ID, URI, Source, Title, Text string }
type Source interface{ Fetch(ctx context.Context) ([]RawDoc, error) }
func DocID(r RawDoc) string
func ToDoc(r RawDoc) Doc
type DirSource struct { Root string; Exts []string }
type CrawlerConfig struct { Seeds, AllowHosts []string; MaxPages, MaxDepth int; RPS float64; UserAgent string; MaxBytes int64
	Clock Clock; Resolver func(context.Context, string) ([]netip.Addr, error); AddrAllowed func(netip.Addr) bool }
func NewCrawler(cfg CrawlerConfig) *Crawler // Fetch(ctx); Refused() []Refusal
func PublicAddr(a netip.Addr) bool

package chunk // import "tinyllm/agent/rag/chunk"
type Tokenizer interface{ Count(ctx context.Context, text string) (int, error) }
type Bytes struct{}
type HTTPTokenizer struct { BaseURL, Model string; Client *http.Client }
type Chunk struct { ID, DocID string; Index int; Text string; Tokens int; Fingerprint string }
type TokenChunker struct { Tok Tokenizer; MaxTokens int } // Chunk(ctx, source.Doc) ([]Chunk, error)
func Fingerprint(text string) string

package embed // import "tinyllm/agent/rag/embed"
type Embedder interface{ Embed(ctx context.Context, texts []string) ([][]float32, error) }
type HTTPEmbedder struct { BaseURL, APIKey, Model string; BatchSize int; Client *http.Client; Retry provider.RetryPolicy }

package store // import "tinyllm/agent/rag/store"
func Open(dir string, m Meta) (*Index, error)
func (x *Index) Entries() []Entry
func (x *Index) Upsert(docID string, es []Entry) error
func (x *Index) Save(now time.Time) (bool, error)
func Ingest(ctx context.Context, src source.Source, ch chunk.Chunker, em embed.Embedder, x *Index) (Stats, error)
```

The catalog writes the chunker as `Chunker.Chunk(d Doc) []Chunk`. Here it takes a `ctx` and returns an error, because the tokenizer is a network call. The resolver and the address rule are injected into `CrawlerConfig` so that tests can fake DNS and exercise every address class without a network. The solution uses only the standard library and `tinyllm/agent/provider`.

## Building it step by step

Before reading the code, it is worth seeing how each design choice follows from a failure you want to prevent.

1. **A chunk the engine truncates loses its end silently**, so every chunk must fit the embedding model's context, measured by that model's tokenizer. Token counts are not additive (BPE merges across a join; some tokenizers charge for separators), so the only correct test is to count the joined candidate.
2. **Prefer natural boundaries.** A piece that is too big is split at the largest unit that fits: paragraph, sentence, word, then character. Characters, not bytes, because a tokenizer rejects or mangles invalid UTF-8.
3. **Greedy packing is enough.** Each piece either extends the current chunk or starts a new one. With an inclusive budget this never produces an over-budget chunk, and it keeps chunk ids stable for unchanged text.
4. **Re-ingest must be cheap.** A chunk's identity for embedding purposes is its text, so its fingerprint is the SHA-256 of the text. Equal fingerprint lists mean an unchanged document; otherwise only new fingerprints need the embedder.
5. **Cosine similarity should be a dot product at query time.** Normalizing every vector once at store time moves the cost to ingest, where it is paid once per chunk.
6. **A crawler lets documents choose where the server connects.** The defense needs checks at the two places the target is decided: the URL's host (allowlist, exact match) and the address actually dialed (all resolved addresses public, dial the checked address). Redirects re-enter both.

### Packing

`Chunk` first turns the document into pieces with their separators, then packs greedily, counting each candidate whole:

```go
	for _, para := range paraRE.Split(d.Text, -1) { // paragraphs: blank-line separated
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		parts, err := c.fit(ctx, para) // one piece, or sentences/words/characters
		// ...
		for i, p := range parts {
			sep := " "
			if i == 0 {
				sep = "\n\n" // the first piece of a paragraph joins with a paragraph break
			}
			pieces = append(pieces, piece{p, sep})
		}
	}
	var out []Chunk
	emit := func(text string, n int) {
		// ids from 0; the fingerprint is of the exact text stored
		out = append(out, Chunk{ID: d.ID + "#" + strconv.Itoa(len(out)), DocID: d.ID, Index: len(out), Text: text, Tokens: n, Fingerprint: Fingerprint(text)})
	}
	cur, curN := "", 0
	for _, p := range pieces {
		if cur == "" {
			// ... cur, curN = p.text, Count(p.text)
			continue
		}
		cand := cur + p.sep + p.text
		n, err := c.Tok.Count(ctx, cand) // count the real joined text, never curN + Count(p)
		// ...
		if n <= c.MaxTokens { // inclusive budget
			cur, curN = cand, n
			continue
		}
		emit(cur, curN)
		cur = p.text
		// ... curN = Count(cur)
	}
	if cur != "" {
		emit(cur, curN)
	}
```

### Splitting

`fit` recurses. A text over budget is split into sentences if it has more than one, else into words, else into characters, and each part is fit again:

```go
func (c TokenChunker) fit(ctx context.Context, text string) ([]string, error) {
	n, err := c.Tok.Count(ctx, text)
	// ...
	if n <= c.MaxTokens {
		return []string{text}, nil
	}
	var parts []string
	switch {
	case len(sentenceRE.FindAllString(text, -1)) > 1: // `[^.!?]+(?:[.!?]+|$)`
		// ... trimmed sentences
	case len(strings.Fields(text)) > 1:
		parts = strings.Fields(text)
	default:
		return c.hardSplit(ctx, text) // one word over budget
	}
	// ... out = fit(part) for each part
}
```

`hardSplit` binary-searches over character boundaries (the indexes `range text` yields), so each cut is the longest UTF-8-valid prefix that fits. A single character that is itself over budget is emitted whole rather than looping forever. For a 60-byte word made of `é` at budget 7, the longest fitting prefix is three 2-byte characters, 6 bytes, so every piece is 6 bytes of valid UTF-8.

### Dedup in `Ingest`

If all fingerprints are equal, skip the document and the index stays clean. Otherwise reuse stored vectors by fingerprint and embed only the rest:

```go
		old := x.docEntries(d.ID)
		if len(old) == len(chunks) {
			same := true
			for i := range chunks {
				same = same && old[i].Fingerprint == chunks[i].Fingerprint
			}
			if same {
				st.Unchanged++
				continue // nothing embedded, nothing marked dirty
			}
		}
		known := map[string][]float32{}
		for _, e := range old {
			known[e.Fingerprint] = e.Vector
		}
		var need []string
		var needAt []int
		es := make([]Entry, len(chunks))
		for i, c := range chunks {
			es[i] = Entry{ /* ... */ Fingerprint: c.Fingerprint, Vector: known[c.Fingerprint]}
			if es[i].Vector == nil {
				need = append(need, c.Text) // only new text costs an embedding
				needAt = append(needAt, i)
			}
		}
		// ... vecs := em.Embed(ctx, need); es[needAt[k]].Vector = vecs[k]
		if err := x.Upsert(d.ID, es); err != nil {
			return st, err
		}
```

### Normalize on write, replace in place

`Upsert` L2-normalizes in float64 and refuses a zero vector, which has no direction. It then puts the document's new chunks where its old ones were, so line order stays stable:

```go
	for i := range es {
		v := es[i].Vector
		// ... dimension check against Meta.Dim
		var ss float64
		for _, f := range v {
			ss += float64(f) * float64(f)
		}
		norm := math.Sqrt(ss)
		if norm == 0 {
			return fmt.Errorf("store: %s has a zero vector", es[i].ChunkID)
		}
		// ... nv[j] = float32(float64(f) / norm)
	}
	// ... rebuild x.entries with es at the document's first old position (or at the end)
	x.dirty = true
```

`Save` returns `(false, nil)` when `!x.dirty`. Otherwise it writes `docs.jsonl`, `vectors.f32` (little-endian float32, no header), and `meta.json`, each through a `.tmp` file, fsync, and rename. The full code is in [`chunk.go`](../../../../course/solutions/go/agent/rag/chunk/chunk.go) and [`store.go`](../../../../course/solutions/go/agent/rag/store/store.go).

### The embedder places vectors by `index`

A missing, out-of-range, or repeated index is an error rather than a silently shifted row:

```go
	if len(out.Data) != n {
		return nil, fmt.Errorf("embed: %d embeddings for %d inputs", len(out.Data), n)
	}
	vecs := make([][]float32, n)
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= n || vecs[d.Index] != nil {
			return nil, fmt.Errorf("embed: bad or repeated index %d", d.Index)
		}
		vecs[d.Index] = d.Embedding // the server may answer in any order
	}
```

A non-200 answer becomes a `provider.APIError` (carrying `Retry-After` when the server sent one), and `batch` retries while `provider.Retryable(err)` holds, sleeping `pol.Delay(attempt, err)`. The full code is in [`embed.go`](../../../../course/solutions/go/agent/rag/embed/embed.go).

### The address rule

`PublicAddr` calls `Unmap` first, so `::ffff:10.0.0.1` is judged as `10.0.0.1`, and then rejects every non-public class:

```go
func PublicAddr(a netip.Addr) bool {
	a = a.Unmap()
	return a.IsValid() && !a.IsLoopback() && !a.IsPrivate() && !a.IsLinkLocalUnicast() &&
		!a.IsLinkLocalMulticast() && !a.IsInterfaceLocalMulticast() && !a.IsMulticast() &&
		!a.IsUnspecified() && !cgnat.Contains(a) && !nat64.Contains(a) && // 100.64/10, 64:ff9b::/96
		!(a.Is4() && a.As4()[0] == 0) // 0/8
}
```

### The dial-time guard

The crawler's `http.Transport` dials only through `dialGuarded`, which resolves the name itself, rejects the name if any address is not allowed, and connects to the address it checked:

```go
func (c *Crawler) dialGuarded(ctx context.Context, network, hostport string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(hostport)
	// ...
	var addrs []netip.Addr
	if a, err := netip.ParseAddr(host); err == nil {
		addrs = []netip.Addr{a} // an IP literal is checked too
	} else if addrs, err = c.cfg.Resolver(ctx, host); err != nil {
		return nil, err
	}
	// ... no addresses: error
	for _, a := range addrs {
		if !c.cfg.AddrAllowed(a) { // ANY bad address refuses the name
			return nil, fmt.Errorf("%w: %s resolves to %s", ErrBlockedAddr, host, a)
		}
	}
	var d net.Dialer
	// Dial the checked address, not the name: no second DNS answer (rebinding).
	return d.DialContext(ctx, network, net.JoinHostPort(addrs[0].Unmap().String(), port))
}
```

`NewCrawler` sets `Proxy: nil` on the transport, since a proxy would make the dial check meaningless, and `CheckRedirect: c.checkRedirect`, which allows at most 5 redirects and checks each target with `hostAllowed`: an exact, case-insensitive name match, for `http` and `https` only.

### Per-host rate

`wait` books the host's next slot under the lock and sleeps outside it, so two hosts never wait for each other:

```go
func (c *Crawler) wait(host string) {
	gap := time.Duration(float64(time.Second) / c.cfg.RPS)
	c.mu.Lock()
	now := c.cfg.Clock.Now()
	at := c.next[host]
	if at.Before(now) {
		at = now // idle host: go now
	}
	c.next[host] = at.Add(gap) // book the slot after this one
	c.mu.Unlock()
	if d := at.Sub(now); d > 0 {
		c.cfg.Clock.Sleep(d)
	}
}
```

### robots.txt

`robotsFor` caches rules per host. A 2xx is parsed, a 4xx allows everything, and a 5xx or a transport error leaves a rule set that allows nothing (RFC 9309: unreachable means assume complete disallow). `allowed` keeps the longest matching rule, with `Allow` winning a tie:

```go
	best, allow := -1, true // no matching rule: allowed
	for _, x := range r.rules {
		if x.re.MatchString(path) && (x.n > best || (x.n == best && x.allow)) {
			best, allow = x.n, x.allow
		}
	}
	return allow
```

`parseRobots` turns each pattern into an anchored regexp (`*` becomes `.*`, a trailing `$` is kept as an anchor) and prefers the group naming our user-agent token over `*`. The full code is in [`crawler.go`](../../../../course/solutions/go/agent/rag/source/crawler.go).

### Why this design

Cost first. Packing calls the tokenizer once per piece, plus once per new chunk, so $O(P)$ tokenizer calls for $P$ pieces; `hardSplit` adds $O(\log L)$ calls per over-long word of $L$ characters. Over HTTP those calls are the dominant cost, which is why the dedup path skips all downstream work for unchanged documents. `Upsert` is $O(n)$ in index size per document, which is acceptable for an in-memory index of thousands of chunks.

Counting the candidate beats summing counts. Summing is cheaper (one cached call per piece) but wrong in both directions: overcounting wastes context, and undercounting lets the engine truncate. Correctness wins. Likewise, fingerprints are taken over the exact text rather than a normalized one: normalizing (lowercasing, collapsing spaces) would treat a meaningful edit as unchanged and keep a stale vector for text that differs from what is stored. And resolving inside the dialer beats pre-checking the URL, because a check on the name before `http.Get` resolves it again is a time-of-check to time-of-use race, exactly the gap DNS rebinding exploits.

Some tradeoffs are accepted on purpose. There is no chunk overlap (`overlap` stays 0 in `meta.json`). A document that disappears from the source is not deleted from the index. HTML is stripped with regexps, which is adequate for documentation pages but is not a full HTML parser.

## What can go wrong

A correct solution must pass the following checks. Together they say what "correct" means for each part of the pipeline and why it matters downstream:

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | three chunks of 60, 45, 28 tokens with ids and fingerprints | the reference packing |
| split order | unit | sentences before words before characters | chunks end at sentence boundaries when they can |
| the budget is never exceeded | property | 40 documents, a separator-charging tokenizer: every chunk within budget, no word lost or repeated | the engine never truncates a chunk |
| a hard split stays UTF-8 | boundary | a 60-byte word of `é` at budget 7: 6-byte valid pieces | no invalid UTF-8 reaches the tokenizer |
| the HTTP tokenizer | conformance | the count is the length of `/v1/tokenize`'s ids | the budget is the embedding model's |
| embedder batches and order | unit | batches of 2, vectors placed by index; wrong count and mixed dimensions are errors | rows of `vectors.f32` match lines of `docs.jsonl` |
| embedder retry | fault | a 503 is retried after `Initial` | ingest survives a busy engine |
| ingest dedup | unit | unchanged: nothing embedded or written; one changed chunk: one embedding | nightly re-ingest is cheap |
| the index format | conformance | `meta.json`, `docs.jsonl` ids and fingerprints, `vectors.f32` size and unit rows, reload | retrieval and context assembly read these files |
| `ToDoc` | unit | script and style dropped, block breaks, entities, titles, doc ids | embeddings of text, not markup |
| the crawler fetches | unit | breadth-first order, fragments, allowlist (lookalike included), robots, refusal reasons | the practice crawler, as a source |
| per-host rate limit | unit | at RPS 2, 500 ms between requests to one host; two hosts finish at 2 s of fake time | polite, without serializing hosts |
| robots.txt | unit | the UA group, longest match, `*` and `$`, 404 and 503 | RFC 9309 |
| SSRF | fault | private, link-local, loopback, IP literals, mapped addresses, a name with one private address, redirects | a document cannot aim the server |
| the public-address table | unit | the address table, including mapped, NAT64, CGNAT | the forms a string check misses |

The mistakes people actually make cluster into a few families. In the chunker, the classic errors are summing counts instead of counting the candidate, an exclusive budget, splitting into words before sentences, and counting bytes instead of tokens. The symptoms are chunks over budget that the engine truncates, and chunks cut mid-sentence; the solution counts `cur + sep + p` with `<=`, tries sentences before words in `fit`, and uses the model's tokenizer through `HTTPTokenizer`. Cutting a long word at a byte offset produces invalid UTF-8 that the tokenizer rejects or mangles; `hardSplit` cuts only at character boundaries. Numbering chunk ids from 1, or fingerprinting a normalized text, makes citations point at the wrong chunk and dedup compare the wrong hash; ids are `<doc_id>#<len(out)>` from 0, and `Fingerprint` hashes the exact chunk text.

> **Warning:** Re-embedding unchanged documents, re-embedding every chunk of a changed one, or rewriting an unchanged index makes the nightly job re-pay for the whole corpus. `Ingest` skips equal fingerprint lists and reuses vectors by fingerprint, and `Save` writes only when dirty.

In the store and embedder, vectors stored unnormalized mean dot products are not cosines and long texts win every search; `Upsert` L2-normalizes every vector. Embeddings placed by response order attach vectors to the wrong chunks, and an embedder without retry dies on one 503; `post` places by `index` and validates it, and `batch` retries with the provider policy.

> **Warning:** Allowing link-local addresses, failing to unmap IPv4-mapped addresses, checking only the first resolved address, or following redirects off the allowlist lets a document make your server fetch the metadata service, an intranet host, or localhost. `PublicAddr` unmaps and rejects every non-public class, `dialGuarded` checks all addresses, and `checkRedirect` reapplies the allowlist.

The crawler has quieter failure modes too. An allowlist matched by prefix crawls lookalike hosts, and keeping fragments fetches the same page twice; `hostAllowed` compares whole names and `push` clears the fragment before the seen check. One rate slot shared by every host makes the crawl slow, while booking slots without waiting hammers a site; `wait` keeps `next` per host and sleeps until the booked slot. Reading a 5xx robots.txt as "allow all", letting the first rule win, or ignoring the UA group fetches pages a site excluded; `robotsFor` disallows on 5xx and errors, `allowed` takes the longest match, and `parseRobots` prefers the UA group. Finally, leaving script and style in the text gives you embeddings of tracking code; `ToDoc` drops `<script>`, `<style>`, and `<head>` content before stripping tags.

## Where this leads

Looking back, the embedder depends on `provider.Retryable` and `RetryPolicy.Delay` from [Types, an OpenAI-compatible provider, and a retry wrapper](../13-agent-sdk/01-types-and-provider.md). Looking forward, [Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR](07-retrieval.md) reads `docs.jsonl` and `vectors.f32`, writes `bm25.idx`, and embeds queries with the same model. [Tools, a registry, and JSON Schema argument validation](../13-agent-sdk/02-tools-and-schemas.md) is background reading only: the RAG tool `search_docs`, built in [Context Assembly, Citations, and the search_docs Tool](08-context-assembly.md), plugs into the registry, but ingest does not. The tokenizer behind the engine's `/v1/tokenize` is the one from [Fast BPE in Rust](../../build-an-llm/p01-tokenizers/05-rust-fast-bpe.md). Together, ingest, retrieval, and context assembly give the agent cited answers over the course docs.

Each piece of this solution has a production counterpart worth studying:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `TokenChunker` | saige semantic chunker | boundaries where the embedding of consecutive sentences changes, not only at punctuation | saige `rag/chunker/semantic` |
| `Ingest` | saige ingest pipeline | parallel stages, retries per document, deletion of documents gone from the source | saige `rag/internal/pipeline/pipeline.go` |
| `Index` | saige `pgstore`, pgvector | an index in Postgres with transactions and filters | saige `rag/pgstore`; [pgvector](https://github.com/pgvector/pgvector) (free) |
| SSRF guard | Smokescreen | an egress proxy that applies the same address rules to every service | [stripe/smokescreen](https://github.com/stripe/smokescreen) (free) |
| `Crawler` | Colly, Scrapy | concurrency per domain, caching, sitemaps | [gocolly/colly](https://github.com/gocolly/colly) (free) |

### Further reading

- OWASP, [Server-Side Request Forgery Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) (free).
- [RFC 9309, Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309) (free).

## Key terms

- **Source**: a component that returns raw documents, such as `DirSource` for files or the crawler for web pages.
- **Token budget**: the maximum token count of a chunk (`MaxTokens`), measured by the embedding model's tokenizer and inclusive.
- **Greedy packing**: extending the current chunk with each piece while the joined candidate fits, emitting it otherwise.
- **Fingerprint**: the hex SHA-256 of a chunk's exact text, used to detect unchanged chunks and reuse their vectors.
- **SSRF (server-side request forgery)**: an attack in which input chooses the address a server connects to.
- **DNS rebinding**: returning a safe address when a name is checked and an internal one when it is resolved again for the connection.
- **IPv4-mapped IPv6 address**: an IPv6 form such as `::ffff:10.0.0.1` that denotes an IPv4 address and must be judged as one.

## Summary

- Chunks are packed to a token budget counted by the embedding model's own tokenizer, and every candidate is counted whole, because token counts do not add up across a join.
- An over-long paragraph is cut at the largest unit that fits: sentences, then words, then characters, never inside a UTF-8 character.
- Ingest is idempotent by fingerprint: an unchanged document embeds and writes nothing, and a changed one re-embeds only its new chunks.
- Vectors are placed by `index`, L2-normalized on write, and saved atomically, so retrieval's cosine is a dot product over rows that match their chunks.
- A crawler is an SSRF risk: hosts are allowlisted exactly, every resolved address is checked at dial time, and redirects are checked like new requests.
- A polite crawler honors robots.txt per RFC 9309 and a per-host rate on a fakeable clock.

## Review questions

1. In the hand example, why is S1 joined to P1 with `\n\n` and not with a space?
   <details><summary>Answer</summary>S1 is the first piece of paragraph 2, so its separator is a paragraph break. Only later pieces of the same paragraph (S2) join with a space. Keeping the original structure also keeps the count honest: the candidate is exactly 36 + 2 + 22 = 60 bytes.</details>
2. Why is `T(c + s + p)` counted instead of `T(c) + T(p)`?
   <details><summary>Answer</summary>Token counts are not additive. A BPE tokenizer can merge characters across the join (the sum overcounts), and a tokenizer that charges for the separator or adds a per-text marker undercounts. Only the real tokenizer on the real candidate gives the number the engine will see.</details>
3. A name resolves to one public and one private address. Why refuse it instead of dialing the public one?
   <details><summary>Answer</summary>Either answer may be returned on a later lookup, and a mixed answer is a typical rebinding setup. A name whose resolution includes an internal address is not trustworthy as a whole; refusing it costs at most one page.</details>
4. Why normalize vectors when storing rather than at query time?
   <details><summary>Answer</summary>Retrieval compares one query against every stored row. With unit rows, cosine similarity is a dot product, so the per-query cost has no norms in it; normalization is paid once per chunk at ingest.</details>
5. What would go wrong if the embedder placed vectors in response order?
   <details><summary>Answer</summary>A server may answer in any order. Vectors would be attached to the wrong chunks, and every search would return passages whose embedding belongs to other text, with nothing in the files to reveal it.</details>
6. Why does the crawler's transport set `Proxy: nil`?
   <details><summary>Answer</summary>With a proxy, the crawler's dialer connects to the proxy, not to the target, so the dial-time address check would vet the proxy's address and the proxy would resolve and connect to whatever the URL named. Disabling the proxy keeps the checked address and the connected address the same.</details>
