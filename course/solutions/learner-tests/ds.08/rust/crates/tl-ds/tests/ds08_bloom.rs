//! Reference learner tests for ds.08 (rung R4: properties with proptest,
//! craft.04). The worked example pins the format; the properties pin the
//! guarantees. Deterministic: a fixed proptest seed and no failure files.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use tl_ds::bloom::{optimal_k, optimal_m, Bloom, BloomError};

fn cfg() -> Config {
    Config { cases: 64, rng_seed: RngSeed::Fixed(5), failure_persistence: None, ..Config::default() }
}

#[test]
fn worked_example_bytes() {
    let mut b = Bloom::with_rate(4, 0.1).unwrap();
    assert_eq!((b.m(), b.k()), (20, 3));
    b.insert(b"cat");
    b.insert(b"dog");
    let bytes = b.to_bytes();
    assert_eq!(&bytes[32..], &[0x09, 0x68, 0x02]);
    assert_eq!(&bytes[..4], b"TLBF");
    assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 2);
    assert!(!b.contains(b"bird"));
}

#[test]
fn sizing() {
    assert_eq!((optimal_m(1000, 0.01), optimal_k(9586, 1000)), (9586, 7));
    assert_eq!((optimal_m(10, 0.3), optimal_k(26, 10)), (26, 2));
}

proptest! {
    #![proptest_config(cfg())]

    #[test]
    fn inserted_items_are_present(items in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..20), 1..200)) {
        let mut b = Bloom::with_rate(items.len() as u64, 0.01).unwrap();
        for it in &items {
            b.insert(it);
        }
        for it in &items {
            prop_assert!(b.contains(it));
        }
    }

    #[test]
    fn bytes_roundtrip(items in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..8), 0..50), n in 1u64..100) {
        let mut b = Bloom::with_rate(n, 0.05).unwrap();
        for it in &items {
            b.insert(it);
        }
        let c = Bloom::from_bytes(&b.to_bytes()).unwrap();
        prop_assert_eq!(c.to_bytes(), b.to_bytes());
        let mut long = b.to_bytes();
        long.push(0);
        prop_assert!(matches!(Bloom::from_bytes(&long), Err(BloomError::Format(_))));
    }

    #[test]
    fn union_equals_one_filter(a in prop::collection::vec(any::<u32>(), 0..50), b in prop::collection::vec(any::<u32>(), 0..50)) {
        let mut fa = Bloom::with_rate(100, 0.02).unwrap();
        let mut fb = Bloom::with_rate(100, 0.02).unwrap();
        let mut both = Bloom::with_rate(100, 0.02).unwrap();
        for x in &a {
            fa.insert(&x.to_le_bytes());
            both.insert(&x.to_le_bytes());
        }
        for x in &b {
            fb.insert(&x.to_le_bytes());
            both.insert(&x.to_le_bytes());
        }
        fa.union(&fb).unwrap();
        prop_assert_eq!(fa.to_bytes(), both.to_bytes());
    }
}

#[test]
fn false_positive_rate_is_near_design() {
    let mut b = Bloom::with_rate(1000, 0.01).unwrap();
    for i in 0..1000u32 {
        b.insert(&i.to_le_bytes());
    }
    let fp = (1_000_000u32..1_050_000).filter(|i| b.contains(&i.to_le_bytes())).count();
    assert!(fp < 1000, "{fp} false positives in 50,000");
}

#[test]
fn malformed_bytes_and_parameters_are_errors() {
    let good = Bloom::with_rate(4, 0.1).unwrap().to_bytes();
    let mut version = good.clone();
    version[4] = 2;
    let mut reserved = good.clone();
    reserved[20] = 1;
    for bad in [version, reserved] {
        assert!(matches!(Bloom::from_bytes(&bad), Err(BloomError::Format(_))));
    }
    assert!(Bloom::new(8, 0).is_err());
    let mut a = Bloom::new(64, 3).unwrap();
    assert!(matches!(a.union(&Bloom::new(64, 4).unwrap()), Err(BloomError::Mismatch(_))));
}
