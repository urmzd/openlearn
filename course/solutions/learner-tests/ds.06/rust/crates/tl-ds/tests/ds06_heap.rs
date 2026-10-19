//! Model test suite for ds.06, written test-first. They go through the
//! public API only (an integration test), and pin the tie-breaking rule.

use std::cmp::Ordering;

use tl_ds::heap::{Heap, LazyHeap};

fn by_key(a: &(u32, u32), b: &(u32, u32)) -> Ordering {
    a.0.cmp(&b.0)
}

#[test]
fn pops_in_order() {
    let mut h: Heap<i32, fn(&i32, &i32) -> Ordering> = Heap::new_min();
    for x in [5, 3, 8, 1, 9, 2] {
        h.push(x);
    }
    assert_eq!(h.peek(), Some(&1));
    assert_eq!(h.into_sorted_vec(), vec![1, 2, 3, 5, 8, 9]);
}

#[test]
fn array_after_pushes() {
    let mut h: Heap<i32, fn(&i32, &i32) -> Ordering> = Heap::new_min();
    for x in [5, 3, 8, 1, 9, 2] {
        h.push(x);
    }
    let a: Vec<i32> = h.array().into_iter().copied().collect();
    assert_eq!(a, vec![1, 3, 2, 5, 9, 8]);
    h.pop();
    let a: Vec<i32> = h.array().into_iter().copied().collect();
    assert_eq!(a, vec![2, 3, 8, 5, 9]);
}

#[test]
fn ties_are_fifo() {
    let mut h = Heap::new(by_key);
    for (i, k) in [1u32, 0, 1, 0].into_iter().enumerate() {
        h.push((k, i as u32));
    }
    let out: Vec<u32> = std::iter::from_fn(|| h.pop()).map(|p| p.1).collect();
    assert_eq!(out, vec![1, 3, 0, 2]);
}

#[test]
fn from_vec_sorts() {
    let h = Heap::from_vec(vec![4, 1, 3, 0, 2], |a: &i32, b: &i32| a.cmp(b));
    assert_eq!(h.into_sorted_vec(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn removed_items_never_pop() {
    let mut h = LazyHeap::new(|a: &i32, b: &i32| a.cmp(b));
    let a = h.push(5);
    let b = h.push(3);
    h.push(8);
    assert!(h.remove(b));
    assert_eq!(h.pop(), Some(5));
    let c = h.push(4);
    assert!(!h.remove(b), "a stale handle must not remove 4");
    assert!(!h.remove(a));
    assert!(h.contains(c));
    assert_eq!(h.pop(), Some(4));
    assert_eq!(h.pop(), Some(8));
    assert_eq!(h.pop(), None);
    assert!(h.is_empty());
}

#[test]
fn many_removes_stay_bounded() {
    let mut h = LazyHeap::new(|a: &u32, b: &u32| a.cmp(b));
    let hs: Vec<_> = (0..1000u32).map(|x| h.push(x)).collect();
    for hd in &hs[1..] {
        h.remove(*hd);
    }
    assert!(h.stale_len() <= h.len() + 32);
    assert_eq!(h.pop(), Some(0));
}
