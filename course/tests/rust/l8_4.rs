//! L8.4 course tests: the radix prefix cache (tl-engine `prefix`).
//!
//! Annotated exemplars (DESIGN 5.12). Block ids are plain numbers here: the
//! cache stores ids, and the block manager (L10.4) that hands it real pool
//! blocks comes later. The golden test replays a trace recorded from an
//! independent Python model (course/oracle/L8.4/radix_trace.py); the
//! property test simulates an engine sharing a fixed set of block ids
//! between running requests, the cache, and a free list. Random operations
//! come from the PCG32 transcribed below from contracts/spec/pcg32.md (the
//! frozen generator, D35).

use std::collections::{HashMap, HashSet};

use tl_engine::prefix::{BlockId, NodeId, RadixCache};

/// The frozen PCG32 (spec/pcg32.md): `pcg32_srandom_r(seed, seq)`.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut g = Pcg32 {
            state: 0,
            inc: (seq << 1) | 1,
        };
        g.next_u32();
        g.state = g.state.wrapping_add(seed);
        g.next_u32();
        g
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        xs.rotate_right((old >> 59) as u32)
    }
    fn below(&mut self, n: u32) -> u32 {
        let t = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= t {
                return r % n;
            }
        }
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn fixture(rel: &str) -> String {
    let dir = std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol");
    std::fs::read_to_string(format!("{dir}/{rel}")).expect("fixture")
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_shared_system_prompt() {
    // WHY: section 3 by hand, block_size 2. Request A ran 1 2 3 4 5: its two
    //      full blocks (10, 11) go in; token 5 is a partial block and stays
    //      A's. Request B's prompt 1 2 3 4 9 9 reuses both blocks (4
    //      tokens). Request C's prompt 1 2 3 4 is fully cached, but the model
    //      must still run on its last token, so C reuses one block only.
    // KIND: unit
    // CATCHES: s01, s03, s06, m02
    // CHAPTER: L8.4 section 3, Worked example by hand
    let mut c = RadixCache::new(2);
    let ins = c.insert(&[1, 2, 3, 4, 5], &[10, 11]);
    assert!(ins.duplicates.is_empty());
    assert_eq!(c.cached_tokens(), 4);
    assert_eq!(c.cached_blocks(), 2);
    let b = c.match_prefix(&[1, 2, 3, 4, 9, 9]);
    assert_eq!((b.matched_tokens, b.blocks.clone()), (4, vec![10, 11]));
    let cc = c.match_prefix(&[1, 2, 3, 4]);
    assert_eq!((cc.matched_tokens, cc.blocks), (2, vec![10]));
    let miss = c.match_prefix(&[7, 7, 7]);
    assert_eq!((miss.matched_tokens, miss.blocks.len()), (0, 0));
    let s = c.stats();
    assert_eq!((s.lookups, s.query_tokens, s.hit_tokens), (3, 13, 6));
    assert!((c.hit_rate() - 6.0 / 13.0).abs() < 1e-12);
}

#[test]
fn the_last_prompt_token_is_never_covered() {
    // WHY: prefill must compute at least one prompt position to get the
    //      logits of the first new token. A prompt of 1, 2, 3, or 4 cached
    //      tokens (block_size 2) is matched to 0, 0, 2, 2 tokens; an empty
    //      prompt matches nothing and does not panic.
    // KIND: boundary
    // CATCHES: s01
    // CHAPTER: L8.4 section 5, Pitfalls
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4, 5, 6], &[10, 11, 12]);
    for (n, want) in [
        (0usize, 0usize),
        (1, 0),
        (2, 0),
        (3, 2),
        (4, 2),
        (5, 4),
        (6, 4),
        (7, 6),
    ] {
        let toks: Vec<u32> = (1..=n as u32).chain(std::iter::once(99)).take(n).collect();
        assert_eq!(
            c.match_prefix(&toks).matched_tokens,
            want,
            "prompt of {n} tokens"
        );
    }
}

#[test]
#[should_panic]
fn insert_takes_exactly_the_full_blocks() {
    // WHY: blocks[i] holds tokens [i*B, (i+1)*B); passing the partial last
    //      block (3 blocks for 5 tokens at B = 2) would cache a block whose
    //      contents are still growing.
    // KIND: boundary
    // CATCHES: s04
    // CHAPTER: L8.4 section 4
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4, 5], &[10, 11, 12]);
}

#[test]
fn a_prefix_computed_twice_is_kept_once() {
    // WHY: two requests with the same system prompt can prefill it at the
    //      same time with different blocks. The cache keeps the first one's
    //      blocks and hands the second one's back for the shared positions,
    //      so the block manager frees them; passing the cached ids
    //      themselves (as a request that matched does) hands nothing back.
    // KIND: unit
    // CATCHES: s02, s05
    // CHAPTER: L8.4 section 2
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4], &[10, 11]);
    let ins = c.insert(&[1, 2, 3, 4, 7, 8], &[20, 21, 22]);
    assert_eq!(ins.duplicates, vec![20, 21]);
    assert_eq!(c.cached_blocks(), 3);
    let m = c.match_prefix(&[1, 2, 3, 4, 7, 8, 0]);
    assert_eq!(m.blocks, vec![10, 11, 22]);
    let again = c.insert(&[1, 2, 3, 4, 7, 8], &[10, 11, 22]);
    assert!(again.duplicates.is_empty());
    assert_eq!(c.cached_blocks(), 3);
    assert_eq!(c.stats().inserted_blocks, 3);
}

#[test]
fn a_locked_prefix_survives_eviction() {
    // WHY: a running request reads its matched blocks on every decode step.
    //      Locking the node its match ended at pins the whole prefix: an
    //      eviction under pressure frees everything else, and the prefix
    //      becomes evictable when the request unlocks it.
    // KIND: unit
    // CATCHES: s07, s08, s09, s10, m01
    // CHAPTER: L8.4 section 2
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4], &[10, 11]);
    c.insert(&[5, 6], &[20]);
    let m = c.match_prefix(&[1, 2, 3, 4, 9]);
    c.lock(m.node);
    assert_eq!(c.evictable_blocks(), 1);
    assert_eq!(c.evict(10), vec![20]);
    assert_eq!(c.cached_blocks(), 2);
    assert_eq!(c.evict(10), Vec::<BlockId>::new());
    c.unlock(m.node);
    assert_eq!(c.evictable_blocks(), 2);
    assert_eq!(c.evict(1), vec![10, 11]);
    assert_eq!(c.cached_tokens(), 0);
    assert_eq!(c.stats().evicted_blocks, 3);
}

#[test]
fn eviction_frees_the_least_recently_used_first() {
    // WHY: three prompts cached in order A, B, C; a new request hits A, so
    //      under pressure B goes first, then C, and A, the one in use most
    //      recently, last.
    // KIND: unit
    // CATCHES: s09
    // CHAPTER: L8.4 section 2
    let mut c = RadixCache::new(1);
    c.insert(&[1, 1], &[10, 11]);
    c.insert(&[2, 2], &[20, 21]);
    c.insert(&[3, 3], &[30, 31]);
    c.match_prefix(&[1, 1, 5]);
    assert_eq!(c.evict(1), vec![20, 21]);
    assert_eq!(c.evict(2), vec![30, 31]);
    assert_eq!(c.evict(2), vec![10, 11]);
}

#[test]
fn golden_reference_trace() {
    // WHY: the catalog's O test: 300 operations of a recorded workload (four
    //      shared system prompts, locks held by running requests, evictions
    //      under pressure) replayed against the results of an independent
    //      Python model (course/oracle/L8.4/radix_trace.py): every match,
    //      every handed-back duplicate, every eviction in order, and the
    //      cached token count after each step.
    // KIND: golden
    // CATCHES: s01, s02, s05, s06, s07, s08, s09, m02
    // CHAPTER: L8.4 section 4
    let text = fixture("L8.4/radix_trace.txt");
    let mut c: Option<RadixCache> = None;
    let mut nodes: HashMap<usize, NodeId> = HashMap::new();
    let mut op = 0usize;
    for (ln, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (head, want) = match line.split_once(" ; ") {
            Some((h, w)) => (
                h,
                w.split_whitespace()
                    .map(|x| x.parse::<u64>().unwrap())
                    .collect::<Vec<_>>(),
            ),
            None => (line.trim_end_matches(" ;"), Vec::new()),
        };
        let mut parts = head.split_whitespace();
        let kind = parts.next().unwrap();
        let nums: Vec<u64> = parts
            .filter(|x| *x != "/")
            .map(|x| x.parse().unwrap())
            .collect();
        let at = format!("trace line {}: {line}", ln + 1);
        match kind {
            "B" => {
                c = Some(RadixCache::new(nums[0] as usize));
                continue;
            }
            "C" => {
                assert_eq!(c.as_ref().unwrap().cached_tokens() as u64, nums[0], "{at}");
                continue;
            }
            _ => {}
        }
        let cache = c.as_mut().unwrap();
        match kind {
            "M" => {
                let toks: Vec<u32> = nums.iter().map(|&x| x as u32).collect();
                let m = cache.match_prefix(&toks);
                let got: Vec<u64> = std::iter::once(m.matched_tokens as u64)
                    .chain(m.blocks.iter().map(|&b| b as u64))
                    .collect();
                assert_eq!(got, want, "{at}");
                nodes.insert(op, m.node);
            }
            "I" => {
                let (toks, blocks) = head[2..].split_once(" / ").unwrap();
                let toks: Vec<u32> = toks
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let blocks: Vec<u32> = blocks
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let ins = cache.insert(&toks, &blocks);
                let got: Vec<u64> = ins.duplicates.iter().map(|&b| b as u64).collect();
                assert_eq!(got, want, "{at}");
                nodes.insert(op, ins.node);
            }
            "L" => cache.lock(nodes[&(nums[0] as usize)]),
            "U" => cache.unlock(nodes[&(nums[0] as usize)]),
            "E" => {
                let got: Vec<u64> = cache
                    .evict(nums[0] as usize)
                    .iter()
                    .map(|&b| b as u64)
                    .collect();
                assert_eq!(got, want, "{at}");
            }
            other => panic!("{at}: unknown op {other}"),
        }
        op += 1;
    }
    assert!(op > 250, "the trace replayed {op} operations");
}

// ---------------------------------------------------------------------------
// property: an engine's block accounting

#[test]
fn blocks_are_conserved_and_locked_prefixes_stay() {
    // WHY: the catalog's I test, as the block manager (L10.4) will use the
    //      cache. 64 block ids circulate between a free list, running
    //      requests, and the cache over 3,000 seeded steps: requests match,
    //      lock, take new blocks (evicting under pressure), insert, and
    //      finish. Every id is in exactly one place after every step; a
    //      running request's matched blocks are never evicted; right after
    //      an insert, a match of the same tokens plus one finds every full
    //      block; and when all requests finish, evicting everything returns
    //      all 64 ids.
    // KIND: property
    // CATCHES: s02, s05, s06, s07, s08, s09
    // CHAPTER: L8.4 section 4
    const N: u32 = 64;
    const B: usize = 4;
    let mut r = Pcg32::new(ol_seed(), 21);
    let mut c = RadixCache::new(B);
    let mut free: Vec<BlockId> = (0..N).rev().collect();
    // a running request: its lock node, the cached blocks it reads, its own blocks
    struct Req {
        node: NodeId,
        shared: Vec<BlockId>,
        own: Vec<BlockId>,
    }
    let mut running: Vec<Req> = Vec::new();
    let systems: Vec<Vec<u32>> = (0..5)
        .map(|s| {
            (0..(B * (1 + s % 3)) as u32)
                .map(|i| i * 7 + s as u32)
                .collect()
        })
        .collect();
    for _ in 0..3000 {
        if r.below(3) > 0 || running.is_empty() {
            let mut prompt = systems[r.below(5) as usize].clone();
            for _ in 0..r.below(3 * B as u32) {
                prompt.push(100 + r.below(4));
            }
            let m = c.match_prefix(&prompt);
            c.lock(m.node);
            let need = prompt.len() / B - m.blocks.len() + 1; // full blocks past the match, plus the tail
            if free.len() < need {
                for b in c.evict(need - free.len()) {
                    assert!(
                        !running.iter().any(|q| q.shared.contains(&b)),
                        "evicted block {b} a request is reading"
                    );
                    free.push(b);
                }
            }
            if free.len() < need {
                c.unlock(m.node); // no room: the request waits
                continue;
            }
            let own: Vec<BlockId> = (0..need).map(|_| free.pop().unwrap()).collect();
            let full: Vec<BlockId> = m
                .blocks
                .iter()
                .chain(own[..need - 1].iter())
                .copied()
                .collect();
            let ins = c.insert(&prompt, &full);
            assert!(
                ins.duplicates.iter().all(|d| own.contains(d)),
                "only the request's own blocks come back"
            );
            let mut probe = prompt[..prompt.len() / B * B].to_vec();
            probe.push(999);
            assert_eq!(
                c.match_prefix(&probe).matched_tokens,
                prompt.len() / B * B,
                "longest prefix after insert"
            );
            // what the cache now owns leaves the request; duplicates and the tail stay its own
            let stored: HashSet<BlockId> = own[..need - 1]
                .iter()
                .copied()
                .filter(|b| !ins.duplicates.contains(b))
                .collect();
            let keep: Vec<BlockId> = own.into_iter().filter(|b| !stored.contains(b)).collect();
            running.push(Req {
                node: m.node,
                shared: m.blocks,
                own: keep,
            });
        } else {
            let q = running.swap_remove(r.below(running.len() as u32) as usize);
            c.unlock(q.node);
            free.extend(q.own);
        }
        let held: usize = running.iter().map(|q| q.own.len()).sum();
        assert_eq!(
            free.len() + held + c.cached_blocks(),
            N as usize,
            "every block is in exactly one place"
        );
        let mut seen: HashSet<BlockId> = free.iter().copied().collect();
        for q in &running {
            for b in &q.own {
                assert!(seen.insert(*b), "block {b} is in two places");
            }
        }
    }
    for q in running.drain(..) {
        c.unlock(q.node);
        free.extend(q.own);
    }
    free.extend(c.evict(usize::MAX));
    free.sort_unstable();
    assert_eq!(free, (0..N).collect::<Vec<_>>());
    assert_eq!(c.cached_tokens(), 0);
}
