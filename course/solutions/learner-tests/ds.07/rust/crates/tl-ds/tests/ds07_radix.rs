//! My tests for ds.07 (rung R3: written first, red against the stub).

use tl_ds::radix::RadixTree;

fn evict<V>(t: &mut RadixTree<V>, n: usize) -> Vec<V> {
    let mut out = Vec::new();
    t.evict(n, |v| out.push(v));
    out
}

#[test]
fn split_by_hand() {
    let mut t = RadixTree::new(1);
    let (abc, _) = t.insert(&[1, 2, 3], vec!['a', 'b', 'c']);
    let (de, back) = t.insert(&[1, 2, 4, 5], vec!['x', 'y', 'd', 'e']);
    assert_eq!(back, vec!['x', 'y']);
    let mid = t.parent(de).unwrap();
    assert_eq!(
        (t.key(mid), t.values(mid)),
        (&[1u32, 2][..], &['a', 'b'][..])
    );
    assert_eq!((t.key(abc), t.parent(abc)), (&[3u32][..], Some(mid)));
    assert_eq!(t.len(), 5);
}

#[test]
fn match_splits_inside_an_edge() {
    let mut t = RadixTree::new(1);
    t.insert(&[1, 2, 3, 4], vec![1u32, 2, 3, 4]);
    let (n, path) = t.match_prefix(&[1, 2, 9]);
    assert_eq!(n, 2);
    assert_eq!(t.values(path[0]), &[1, 2]);
    let mut g = RadixTree::new(2);
    g.insert(&[1, 2, 3, 4], vec![7u32, 8]);
    assert_eq!(g.match_prefix(&[1, 2, 3, 9]).0, 2);
    assert_eq!(g.match_prefix(&[1, 2, 3, 4]).0, 4);
}

#[test]
fn insert_hands_back_the_stored_part() {
    let mut t = RadixTree::new(1);
    let (a, _) = t.insert(&[5, 6], vec![1u32, 2]);
    let (b, back) = t.insert(&[5, 6], vec![8, 9]);
    assert_eq!((a, back), (b, vec![8, 9]));
    assert_eq!(t.len(), 2);
}

#[test]
fn oldest_leaf_goes_first() {
    let mut t = RadixTree::new(1);
    t.insert(&[1], vec![10u32]);
    t.insert(&[2], vec![20]);
    t.insert(&[3], vec![30]);
    t.match_prefix(&[1]);
    assert_eq!(evict(&mut t, 1), vec![20]);
    assert_eq!(evict(&mut t, 1), vec![30]);
    assert_eq!(evict(&mut t, 1), vec![10]);
}

#[test]
fn locks_pin_ancestors() {
    let mut t = RadixTree::new(1);
    let (leaf, _) = t.insert(&[1, 2, 3], vec![1u32, 2, 3]);
    t.insert(&[1, 2, 7], vec![0, 0, 7]);
    t.lock(leaf);
    let mid = t.parent(leaf).unwrap();
    assert_eq!(t.lock_count(mid), 1);
    assert_eq!(evict(&mut t, 100), vec![7]);
    t.unlock(leaf);
    assert_eq!(evict(&mut t, 100), vec![3, 1, 2]);
}

#[test]
fn parent_becomes_evictable() {
    let mut t = RadixTree::new(1);
    t.insert(&[1, 2, 3], vec![1u32, 2, 3]);
    t.insert(&[1, 2, 4], vec![1, 2, 4]);
    t.insert(&[9], vec![9]);
    assert_eq!(evict(&mut t, 2), vec![3, 4]);
    assert_eq!(evict(&mut t, 1), vec![1, 2]);
    assert_eq!(t.node_count(), 1);
}
