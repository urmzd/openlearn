// Parity driver: kahan_sum (ds.90).
include!("../parity/ol_parity.rs");

fn main() {
    for line in ssp_lines() {
        let xs = ssp_f64s(&line, "xs").expect("xs");
        println!("{:?}", tl_demo::acc::kahan_sum(&xs));
    }
}
