//! My tests for L8.4 (rung R4: properties written as seeded loops).

use tl_engine::prefix::RadixCache;

#[test]
fn a_fully_cached_prompt_matches_all_but_its_last_block() {
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4], &[10, 11]);
    assert_eq!(c.match_prefix(&[1, 2, 3, 4]).blocks, vec![10]);
    assert_eq!(c.match_prefix(&[1, 2, 3, 4, 5]).blocks, vec![10, 11]);
    assert_eq!(c.cached_tokens(), 4);
    let s = c.stats();
    assert_eq!((s.lookups, s.query_tokens, s.hit_tokens), (2, 9, 6));
}

#[test]
fn inserting_what_a_match_returned_hands_nothing_back() {
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3, 4, 5, 6], &[10, 11, 12]);
    let m = c.match_prefix(&[1, 2, 3, 4, 5, 6, 7]);
    let mut blocks = m.blocks.clone();
    blocks.push(13);
    assert!(c
        .insert(&[1, 2, 3, 4, 5, 6, 7, 8], &blocks)
        .duplicates
        .is_empty());
    assert_eq!(c.insert(&[1, 2, 9, 9], &[20, 21]).duplicates, vec![20]);
}

#[test]
#[should_panic]
fn a_partial_block_is_refused() {
    let mut c = RadixCache::new(2);
    c.insert(&[1, 2, 3], &[10, 11]);
}

#[test]
fn a_locked_prefix_is_never_evicted() {
    let mut c = RadixCache::new(1);
    c.insert(&[1, 2], &[10, 11]);
    c.insert(&[3], &[30]);
    let m = c.match_prefix(&[1, 2, 0]);
    c.lock(m.node);
    assert_eq!(c.evictable_blocks(), 1);
    assert_eq!(c.evict(10), vec![30]);
    c.unlock(m.node);
    assert_eq!(c.evict(10), vec![10, 11]);
    assert_eq!(c.stats().evicted_blocks, 3);
}

#[test]
fn blocks_in_equal_blocks_out() {
    let mut c = RadixCache::new(2);
    let mut seed = 7u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let mut given = 0usize;
    let mut back = 0usize;
    let mut id = 0u32;
    for _ in 0..400 {
        let n = 2 * (1 + next() as usize % 4);
        let toks: Vec<u32> = (0..n).map(|_| next() % 3).collect();
        let blocks: Vec<u32> = (0..n / 2)
            .map(|_| {
                id += 1;
                id
            })
            .collect();
        given += blocks.len();
        back += c.insert(&toks, &blocks).duplicates.len();
        if next() % 4 == 0 {
            back += c.evict(1 + next() as usize % 3).len();
        }
        assert_eq!(given, back + c.cached_blocks());
    }
    back += c.evict(usize::MAX).len();
    assert_eq!(given, back);
}
