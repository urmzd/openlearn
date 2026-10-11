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
#[ignore]
fn bench_kahan_1m() {
    // WHY: the course budget for ds.90 (`ol bench ds.90`, release build).
    // KIND: bench
    let xs: Vec<f64> = (0..1_000_000).map(|i| (i % 7) as f64).collect();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        std::hint::black_box(tl_demo::acc::kahan_sum(std::hint::black_box(&xs)));
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("{{\"melem_per_s\": {}}}", 1.0 / best);
}
