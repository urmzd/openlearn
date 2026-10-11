// Parity driver for `bloom`, implementation `rust` (ds.08): an example of the
// harness crate ol-tests. Reads one case per stdin line
// {"n", "p", "insert": [hex], "probe": [hex]} and prints
// {"m", "k", "bytes", "contains"} for the filter tl_ds::bloom::Bloom builds.
include!("../parity/ol_parity.rs");

use tl_ds::bloom::Bloom;

/// The hex strings of the JSON array under `key` (no escapes in hex).
fn hex_list(line: &str, key: &str) -> Vec<Vec<u8>> {
    let s = ssp_find(line, key).and_then(|s| s.strip_prefix('[')).expect("array");
    let body = &s[..s.find(']').expect("]")];
    body.split(',')
        .map(|x| x.trim().trim_matches('"'))
        .filter(|x| !x.is_empty() || body.contains("\"\""))
        .map(|x| (0..x.len()).step_by(2).map(|i| u8::from_str_radix(&x[i..i + 2], 16).unwrap()).collect())
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    for line in ssp_lines() {
        if line.trim().is_empty() {
            continue;
        }
        let n = ssp_u64(&line, "n").expect("n");
        let p = ssp_f64(&line, "p").expect("p");
        let mut b = Bloom::with_rate(n, p).expect("with_rate");
        for it in hex_list(&line, "insert") {
            b.insert(&it);
        }
        let contains: Vec<&str> = hex_list(&line, "probe").iter().map(|x| if b.contains(x) { "true" } else { "false" }).collect();
        println!(
            "{{\"m\":{},\"k\":{},\"bytes\":\"{}\",\"contains\":[{}]}}",
            b.m(),
            b.k(),
            hex(&b.to_bytes()),
            contains.join(",")
        );
    }
}
