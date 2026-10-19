# RAG ingest: crawler with SSRF allowlist, chunker, embedder, store

## Overview

| | |
|---|---|
| **Module** | `ag.06` · build · Go · Pass 10 · reading time 2 h |
| **Solution** | [`go/agent/rag/source/source.go`](../../../../course/solutions/go/agent/rag/source/source.go) (`RawDoc`, `Doc`, `ToDoc`, `DocID`, `DirSource`), [`go/agent/rag/source/crawler.go`](../../../../course/solutions/go/agent/rag/source/crawler.go) (`Crawler`, `PublicAddr`), [`go/agent/rag/chunk/chunk.go`](../../../../course/solutions/go/agent/rag/chunk/chunk.go) (`TokenChunker`, `Bytes`, `HTTPTokenizer`, `Fingerprint`), [`go/agent/rag/embed/embed.go`](../../../../course/solutions/go/agent/rag/embed/embed.go) (`HTTPEmbedder`), [`go/agent/rag/store/store.go`](../../../../course/solutions/go/agent/rag/store/store.go) (`Open`, `Index.Upsert`, `Index.Save`, `Ingest`) |
| **Contract** | the index directory of [`course/contracts/formats/rag-index.md`](../../../../course/contracts/formats/rag-index.md) (`meta.json`, `docs.jsonl`, `vectors.f32`); `/v1/tokenize` and `/v1/embeddings` of [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) |
| **Needs** | [`ag.01` provider](../13-agent-sdk/01-types-and-provider.md) (the embedder retries with `Retryable` and `RetryPolicy`) · reading: [`ag.02` tools](../13-agent-sdk/02-tools-and-schemas.md), the engine's tokenizer (`L1.5`), the crawler warm-up in [practice Go 01](../../../../practice/build/cloud/go/) |
| **Used by** | `ag.07` retrieval reads the index this module writes; together with `ag.07` and `ag.08` it gives the agent cited answers over the course docs |
| **Optional depth** | OWASP, [Server-Side Request Forgery Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) (free); [RFC 9309, Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309) (free) |

## Key Takeaways

- Chunks are packed to a **token budget counted by the embedding model's own tokenizer**, and every candidate chunk is counted whole: token counts do not add up across a join.
- An over-long paragraph is cut at the **largest unit that fits**: sentences, then words, then characters, and never inside a UTF-8 character.
- Ingest is **idempotent by fingerprint**: an unchanged document embeds nothing and writes nothing; a changed one re-embeds only its new chunks.
- A crawler fetches URLs that documents name, so it is an **SSRF** risk: hosts are allowlisted, every resolved address is checked at dial time (no loopback, private, link-local, ...), and a redirect is checked like a new request.
- It is a polite crawler: **robots.txt** per RFC 9309 and a **per-host rate**, measured on a clock that can be faked.

## How to read this chapter

Sections 1 to 3 build the ingest pipeline from first principles and pack one document by hand. Section 4 fixes the package APIs. Section 5 walks through the chunker, the store's dedup, the embedder, and the crawler's SSRF guard, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The agent of `ag.03` knows only what the model memorized. The goal of this pass is an agent that answers questions about the course and its own docs, with citations, which means retrieval: find the passages that answer the question and put them in the context. Retrieval (`ag.07`) searches an index; this module builds that index. Each step has a way to quietly ruin the answers: chunks over the embedding model's context are truncated by the engine and lose their ends, a crawler that follows any link fetches `http://169.254.169.254/latest/meta-data/` (cloud credentials) because a document linked it, a nightly re-ingest that re-embeds everything costs hours, and vectors stored unnormalized make cosine similarity wrong.

## 2. Principles

### 2.1 Sources and text

A **source** returns raw documents: `DirSource` reads files under a directory (`.md` and `.txt` by default, in path order), the **crawler** fetches web pages. `ToDoc` turns a raw document into text: HTML loses `<script>`, `<style>`, and `<head>` content and every tag, block tags (`<p>`, `<li>`, `<h1>`, ...) become paragraph breaks, entities are decoded after tags are removed, spaces collapse, and paragraphs are separated by exactly one blank line. The doc id is the URL without its fragment (scheme and host lowercased) or the file path.

The crawler goes breadth first from its seeds, follows `<a href>` links resolved against the page's final URL, drops `#fragments` (so `/a` and `/a#top` are one page), and visits each URL once, up to `MaxPages` and `MaxDepth`. Two politeness rules:

- **robots.txt** (RFC 9309), fetched once per host: the group naming our user agent applies, else the `*` group; within it the **longest** matching rule wins and `Allow` wins a tie; `*` matches any run and a trailing `$` anchors. A 4xx robots.txt means no restrictions; a 5xx or no answer means the site cannot say, so nothing is fetched.
- **rate**: requests to one host (robots.txt included) are at least $1/\text{RPS}$ apart. Each host has its own next free slot; the crawler sleeps only until that host's slot, so two hosts do not wait for each other. Time comes from a `Clock`, which can be replaced by a fake that advances only on `Sleep`.

### 2.2 Chunking to a token budget

| Symbol | Meaning |
|---|---|
| $B$ | the budget, `MaxTokens` |
| $T(x)$ | the token count of text $x$ by the embedding model's tokenizer |
| $p_1, \dots, p_k$ | the pieces of a document, each with $T(p_i) \le B$ |
| $s_i$ | the separator before $p_i$: `"\n\n"` between paragraphs, `" "` inside one |

Pieces: each paragraph, or, when $T(\text{paragraph}) > B$, its sentences, or a sentence's words, or a word's longest prefixes that fit (cut at a character boundary). Packing is greedy: keep a current chunk $c$; for each piece, if $T(c \,\Vert\, s_i \,\Vert\, p_i) \le B$, extend $c$; otherwise emit $c$ and start a new one with $p_i$.

The count is taken on the **joined candidate**, never as $T(c) + T(p_i)$. A BPE tokenizer can merge across the join (the sum overcounts), and a tokenizer that spends tokens on separators or adds a marker per text undercounts (the sum lets chunks overflow). Only the real tokenizer on the real text is right: `HTTPTokenizer` asks the engine (`/v1/tokenize`, the length of `ids`); `Bytes` is the tracer's byte-level tokenizer. The budget is inclusive.

A chunk's id is `<doc_id>#<k>`, $k$ its position in the document from 0, and its fingerprint is the hex SHA-256 of its text.

### 2.3 Embedding

`HTTPEmbedder` posts batches (64 inputs by default; the contract allows 256) to `/v1/embeddings` and places each vector by its `index` field: a server may answer in any order. An answer with the wrong count, a repeated or out-of-range index, or vectors of different lengths is an error, never a misaligned index. Overload (503, 429) and broken connections are retried with the provider's policy from `ag.01`.

### 2.4 The index and fingerprint dedup

`rag-index.md` fixes the files: `docs.jsonl` (one chunk per line, line $i$ is chunk $i$), `vectors.f32` ($n \times d$ little-endian float32, no header), `meta.json` (`n_chunks`, `dim`, the models, the chunker and BM25 settings). Every row is L2-normalized when stored, so retrieval's cosine similarity is a dot product. `bm25.idx` is written by retrieval (`ag.07`) from the same chunks.

`Ingest` chunks each document and compares the chunk fingerprints with the stored ones for that doc id. All equal: the document is **unchanged**, nothing is embedded and the index is not marked dirty. Otherwise the chunks whose fingerprint the document already had keep their stored vectors, only new ones are embedded, and the document's chunks are replaced in place. `Save` writes nothing when nothing changed, and otherwise writes each file to a temporary name, fsyncs, and renames, so a reader never sees half a file.

### 2.5 Server-side request forgery

A crawler is a program that makes HTTP requests to addresses chosen by its input. If the input is a document an attacker wrote, the attacker chooses where the server connects: an admin port on localhost, a database on the private network, the cloud metadata service at `169.254.169.254`. Three checks, in this order:

1. **Host allowlist**, on the URL's host name before any request: exact names only (a prefix match lets `docs.example.evil.example` in).
2. **Address check at dial time**: the crawler resolves the name itself, refuses if **any** address is not public, and dials the address it checked. Checking a name and letting the HTTP library resolve it again opens a race (DNS rebinding: the second answer is `127.0.0.1`). Not public: loopback, private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), link-local (`169.254/16`, `fe80::/10`), unspecified, multicast, carrier-grade NAT (`100.64/10`), NAT64 (`64:ff9b::/96`, which embeds an IPv4 address), and `0/8`. IPv4-mapped IPv6 (`::ffff:10.0.0.1`) is checked as the IPv4 address it is.
3. **Redirects** are new requests: the allowlist is checked again for each target (at most 5), and the target's connection goes through the same dial check.

## 3. Worked example by hand

Document `docs/kv.md`, the byte tokenizer ($T(x)$ = bytes of $x$), budget $B = 60$:

```
Paged attention stores KV in blocks.                                  (36 bytes)

Blocks are fixed size. A block table maps token positions to blocks.  (68 bytes)

Eviction frees whole blocks.                                          (28 bytes)
```

Paragraph 2 alone is $68 > 60$, so it becomes its two sentences, 22 and 45 bytes. Pieces and packing:

| Piece | Separator | Candidate | $T$ | Fits? | Chunk so far |
|---|---|---|---|---|---|
| P1 (36) | | P1 | 36 | yes | P1 |
| S1 (22) | `\n\n` | P1 + `\n\n` + S1 | 60 | yes (inclusive) | P1, S1 |
| S2 (45) | `" "` | ... + `" "` + S2 | 106 | no: emit chunk 0 (60) | S2 |
| P3 (28) | `\n\n` | S2 + `\n\n` + P3 | 75 | no: emit chunk 1 (45) | P3 |
| end | | | | emit chunk 2 (28) | |

Three chunks: `docs/kv.md#0` (60 tokens, paragraph 1 and the first sentence), `#1` (45), `#2` (28), each with the SHA-256 of its text as fingerprint.

Re-ingesting the unchanged file gives the same three fingerprints: nothing is embedded, `Save` writes nothing. Changing the last paragraph to "Eviction frees whole blocks at once." changes only chunk 2's fingerprint: one text is embedded.

## 4. The interface

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

The catalog writes `Chunker.Chunk(d Doc) []Chunk`; here it takes a `ctx` and returns an error, because the tokenizer is a network call. The solution uses only the standard library and `tinyllm/agent/provider`.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: three chunks of 60, 45, 28 tokens with ids and fingerprints | the reference packing |
| split order | unit | sentences before words before characters | chunks end at sentence boundaries when they can |
| the budget is never exceeded | property | 40 documents, a separator-charging tokenizer: every chunk within budget, no word lost or repeated | the engine never truncates a chunk |
| a hard split stays UTF-8 | boundary | a 60-byte word of `é` at budget 7: 6-byte valid pieces | no invalid UTF-8 reaches the tokenizer |
| the HTTP tokenizer | conformance | the count is the length of `/v1/tokenize`'s ids | the budget is the embedding model's |
| embedder batches and order | unit | batches of 2, vectors placed by index; wrong count and mixed dimensions are errors | rows of `vectors.f32` match lines of `docs.jsonl` |
| embedder retry | fault | a 503 is retried after `Initial` | ingest survives a busy engine |
| ingest dedup | unit | unchanged: nothing embedded or written; one changed chunk: one embedding | nightly re-ingest is cheap |
| the index format | conformance | `meta.json`, `docs.jsonl` ids and fingerprints, `vectors.f32` size and unit rows, reload | `ag.07` and `ag.08` read these files |
| `ToDoc` | unit | script and style dropped, block breaks, entities, titles, doc ids | embeddings of text, not markup |
| the crawler fetches | unit | breadth-first order, fragments, allowlist (lookalike included), robots, refusal reasons | the practice crawler, as a source |
| per-host rate limit | unit | at RPS 2, 500 ms between requests to one host; two hosts finish at 2 s of fake time | polite, without serializing hosts |
| robots.txt | unit | the UA group, longest match, `*` and `$`, 404 and 503 | RFC 9309 |
| SSRF | fault | private, link-local, loopback, IP literals, mapped addresses, a name with one private address, redirects | a document cannot aim the server |
| the public-address table | unit | the address table, including mapped, NAT64, CGNAT | the forms a string check misses |

## 5. The solution, step by step

### 5.1 From first principles

1. **A chunk the engine truncates loses its end silently**, so every chunk must fit the embedding model's context, measured by that model's tokenizer. Token counts are not additive (BPE merges across a join; some tokenizers charge for separators), so the only correct test is to count the joined candidate. In the hand example the second candidate is exactly 60 = $B$ only because the join `\n\n` is counted.
2. **Prefer natural boundaries.** A piece that is too big is split at the largest unit that fits: paragraph, sentence, word, then character. Characters, not bytes, because a tokenizer rejects or mangles invalid UTF-8.
3. **Greedy packing is enough.** Each piece either extends the current chunk or starts a new one; with an inclusive budget this never produces an over-budget chunk and keeps chunk ids stable for unchanged text.
4. **Re-ingest must be cheap.** A chunk's identity for embedding purposes is its text, so its fingerprint is SHA-256 of the text. Equal fingerprint lists mean an unchanged document; otherwise only new fingerprints need the embedder.
5. **Cosine similarity should be a dot product at query time.** Normalizing every vector once at store time moves the cost to ingest, where it is paid once per chunk.
6. **A crawler lets documents choose where the server connects.** Defending that needs checks at the two places the target is decided: the URL's host (allowlist, exact match) and the address actually dialed (all resolved addresses public, dial the checked address). Redirects re-enter both.

### 5.2 The code, annotated

**Packing.** `Chunk` first turns the document into pieces with their separators, then packs greedily, counting each candidate whole:

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

**Splitting.** `fit` recurses: a too-long text is split into sentences if it has more than one, else words, else characters, and each part is fit again:

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

`hardSplit` binary-searches over character boundaries (the indexes `range text` yields), so each cut is the longest UTF-8-valid prefix that fits; a single character over budget is emitted whole rather than looping forever. For the `é` word at budget 7, the longest fitting prefix is three 2-byte characters: 6 bytes.

**Dedup in `Ingest`.** All fingerprints equal: skip, and the index stays clean. Otherwise reuse stored vectors by fingerprint and embed only the rest:

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

**Normalize on write, replace in place.** `Upsert` L2-normalizes in float64 (and refuses a zero vector, which has no direction), then puts the document's new chunks where its old ones were, so line order stays stable:

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

`Save` returns `(false, nil)` when `!x.dirty`, and otherwise writes `docs.jsonl`, `vectors.f32` (little-endian float32, no header), and `meta.json`, each through a `.tmp` file, fsync, and rename. Full solutions: [`chunk.go`](../../../../course/solutions/go/agent/rag/chunk/chunk.go), [`store.go`](../../../../course/solutions/go/agent/rag/store/store.go).

**The embedder places vectors by `index`.** A missing, out-of-range, or repeated index is an error rather than a silently shifted row:

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

A non-200 answer becomes a `provider.APIError` (with `Retry-After` when sent), and `batch` retries while `provider.Retryable(err)` holds, sleeping `pol.Delay(attempt, err)`. Full solution: [`embed.go`](../../../../course/solutions/go/agent/rag/embed/embed.go).

**The address rule.** `Unmap` first, so `::ffff:10.0.0.1` is judged as `10.0.0.1`; then every non-public class:

```go
func PublicAddr(a netip.Addr) bool {
	a = a.Unmap()
	return a.IsValid() && !a.IsLoopback() && !a.IsPrivate() && !a.IsLinkLocalUnicast() &&
		!a.IsLinkLocalMulticast() && !a.IsInterfaceLocalMulticast() && !a.IsMulticast() &&
		!a.IsUnspecified() && !cgnat.Contains(a) && !nat64.Contains(a) && // 100.64/10, 64:ff9b::/96
		!(a.Is4() && a.As4()[0] == 0) // 0/8
}
```

**The dial-time guard.** The crawler's `http.Transport` dials only through `dialGuarded`, which resolves the name itself, rejects if any address is not allowed, and connects to the address it checked:

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

`NewCrawler` sets `Proxy: nil` on the transport (a proxy would make the dial check meaningless) and `CheckRedirect: c.checkRedirect`, which allows at most 5 redirects and checks each target with `hostAllowed`, an exact, case-insensitive name match for `http` and `https` only.

**Per-host rate.** `wait` books the host's next slot under the lock and sleeps outside it, so two hosts never wait for each other:

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

**robots.txt.** `robotsFor` caches per host: a 2xx is parsed, a 4xx allows everything, and a 5xx or a transport error leaves a rule set that allows nothing (RFC 9309: unreachable means assume complete disallow). `allowed` keeps the longest matching rule with `Allow` winning a tie:

```go
	best, allow := -1, true // no matching rule: allowed
	for _, x := range r.rules {
		if x.re.MatchString(path) && (x.n > best || (x.n == best && x.allow)) {
			best, allow = x.n, x.allow
		}
	}
	return allow
```

`parseRobots` turns each pattern into an anchored regexp (`*` to `.*`, a trailing `$` kept as an anchor) and prefers the group naming our user-agent token over `*`. Full solution: [`crawler.go`](../../../../course/solutions/go/agent/rag/source/crawler.go).

### 5.3 Why this solution

- **Cost.** Packing calls the tokenizer once per piece (plus once per new chunk), so $O(P)$ tokenizer calls for $P$ pieces; `hardSplit` adds $O(\log L)$ calls per over-long word of $L$ characters. Over HTTP that is the dominant cost, which is why the dedup path skips chunking's downstream work entirely for unchanged documents. `Upsert` is $O(n)$ in index size per document, acceptable for an in-memory index of thousands of chunks.
- **Counting the candidate over summing counts.** Summing is cheaper (one call per piece, cached) but wrong in both directions; overcounting wastes context, undercounting lets the engine truncate. Correctness wins.
- **Fingerprints of exact text over normalized text.** Normalizing (lowercase, collapse spaces) would treat a meaningful edit as unchanged and keep a stale vector for text that differs from what is stored.
- **Resolving inside the dialer over pre-checking the URL.** A check on the name before `http.Get` resolves it again is a time-of-check to time-of-use race; DNS rebinding exploits exactly that gap.
- **Accepted tradeoffs.** No chunk overlap (`overlap` stays 0 in `meta.json`); a document gone from the source is not deleted from the index; HTML is stripped with regexps, adequate for documentation pages but not a full HTML parser.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. counts summed instead of counting the candidate, an exclusive budget, words before sentences, bytes instead of tokens | chunks over budget, truncated by the engine; chunks cut mid-sentence | `Chunk` counts `cur + sep + p` with `<=`; `fit` tries sentences before words; `HTTPTokenizer` uses the model's tokenizer |
| 2. long words cut at a byte offset | invalid UTF-8 rejected or mangled by the tokenizer | `hardSplit` cuts only at character boundaries |
| 3. chunk ids from 1, fingerprints of a normalized text | citations point at the wrong chunk; dedup compares the wrong hash | ids are `<doc_id>#<len(out)>` from 0; `Fingerprint` hashes the exact chunk text |
| 4. unchanged documents re-embedded, every chunk of a changed one re-embedded, unchanged indexes rewritten | the nightly job re-pays for the whole corpus | `Ingest` skips equal fingerprint lists and reuses vectors by fingerprint; `Save` writes only when dirty |
| 5. vectors stored unnormalized | dot products are not cosines; long texts win every search | `Upsert` L2-normalizes every vector |
| 6. embeddings placed by response order, no retry | vectors attached to the wrong chunks; ingest dies on one 503 | `post` places by `index` and validates it; `batch` retries with the provider policy |
| 7. link-local allowed, mapped addresses not unmapped, only the first address checked, redirects off the allowlist followed | the metadata service, an intranet host, or localhost fetched on a document's say-so | `PublicAddr` unmaps and rejects every non-public class; `dialGuarded` checks all addresses; `checkRedirect` reapplies the allowlist |
| 8. the allowlist matched by prefix, fragments kept | lookalike hosts crawled; the same page fetched twice | `hostAllowed` compares whole names; `push` clears the fragment before the seen check |
| 9. one rate slot for every host, or slots booked without waiting | a slow crawl, or a site hammered | `wait` keeps `next` per host and sleeps until the booked slot |
| 10. a 5xx robots.txt read as "allow all", the first rule winning, the UA group ignored | pages a site excluded are fetched | `robotsFor` disallows on 5xx and errors; `allowed` takes the longest match; `parseRobots` prefers the UA group |
| 11. script and style left in the text | embeddings of tracking code | `ToDoc` drops `<script>`, `<style>`, and `<head>` content before stripping tags |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `provider.Retryable` and `RetryPolicy.Delay` for the embedder |
| Forward | `ag.07` | reads `docs.jsonl` and `vectors.f32`, writes `bm25.idx`, and embeds queries with the same model |

`ag.02` is reading: the RAG tools (`search_docs`, `ag.08`) plug into the registry, ingest does not. `L1.5` is the tokenizer behind the engine's `/v1/tokenize`.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `TokenChunker` | saige semantic chunker | boundaries where the embedding of consecutive sentences changes, not only at punctuation | saige `rag/chunker/semantic` |
| `Ingest` | saige ingest pipeline | parallel stages, retries per document, deletion of documents gone from the source | saige `rag/internal/pipeline/pipeline.go` |
| `Index` | saige `pgstore`, pgvector | an index in Postgres with transactions and filters | saige `rag/pgstore`; [pgvector](https://github.com/pgvector/pgvector) (free) |
| SSRF guard | Smokescreen | an egress proxy that applies the same address rules to every service | [stripe/smokescreen](https://github.com/stripe/smokescreen) (free) |
| `Crawler` | Colly, Scrapy | concurrency per domain, caching, sitemaps | [gocolly/colly](https://github.com/gocolly/colly) (free) |
