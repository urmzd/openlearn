use tl_demo::acc::kahan_sum;

#[test]
fn hand_example() {
    // WHY: the chapter's worked example: 1 + 2 + 3 = 6 exactly.
    // KIND: unit
    assert_eq!(kahan_sum(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn compensates_small_terms() {
    // WHY: ten terms of 1e-16 vanish when added to 1.0 one at a time
    //      (each is under half an ulp of 1.0); compensation keeps them.
    // KIND: differential
    let mut xs = vec![1.0];
    xs.extend(std::iter::repeat(1e-16).take(10));
    let naive: f64 = xs.iter().sum();
    assert_eq!(naive, 1.0);
    assert!((kahan_sum(&xs) - (1.0 + 1e-15)).abs() <= 2.3e-16);
}

#[test]
fn testkit_clock_and_failpoints_reach_course_tests() {
    // WHY: the Rust testkit is a dependency of ol-tests (fake clock, failpoints).
    // KIND: unit
    use tl_testkit::clock::{Clock, FakeClock};
    let c = FakeClock::default();
    c.sleep(std::time::Duration::from_secs(3));
    assert_eq!(c.now(), std::time::Duration::from_secs(3));
    tl_testkit::failpoint::load("ds/sum=error(boom)").unwrap();
    assert!(tl_testkit::failpoint::inject("ds/sum").is_err());
}
