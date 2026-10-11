// ol_parity.rs: the parity driver kit for Rust (course/DESIGN.md 5.8).
// A driver is an example of the harness crate ol-tests and pulls this in with
//     include!("../parity/ol_parity.rs");
// `ol parity` sends one JSON object per stdin line and reads one JSON value
// per stdout line. These scanners read numbers, number arrays, and strings
// without escapes, which is all the suites send.

#[allow(dead_code)]
pub fn ssp_find<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\"", key);
    let mut from = 0;
    while let Some(i) = line[from..].find(&pat) {
        let rest = line[from + i + pat.len()..].trim_start();
        if let Some(r) = rest.strip_prefix(':') {
            return Some(r.trim_start());
        }
        from += i + 1;
    }
    None
}

#[allow(dead_code)]
pub fn ssp_f64(line: &str, key: &str) -> Option<f64> {
    let s = ssp_find(line, key)?;
    let end = s.find(|c: char| c == ',' || c == '}' || c == ']').unwrap_or(s.len());
    s[..end].trim().parse().ok()
}

#[allow(dead_code)]
pub fn ssp_u64(line: &str, key: &str) -> Option<u64> {
    let s = ssp_find(line, key)?;
    let end = s.find(|c: char| c == ',' || c == '}').unwrap_or(s.len());
    s[..end].trim().parse().ok()
}

#[allow(dead_code)]
pub fn ssp_f64s(line: &str, key: &str) -> Option<Vec<f64>> {
    let s = ssp_find(line, key)?.strip_prefix('[')?;
    let end = s.find(']')?;
    let body = s[..end].trim();
    if body.is_empty() {
        return Some(vec![]);
    }
    body.split(',').map(|x| x.trim().parse().ok()).collect()
}

#[allow(dead_code)]
pub fn ssp_str<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let s = ssp_find(line, key)?.strip_prefix('"')?;
    Some(&s[..s.find('"')?])
}

#[allow(dead_code)]
pub fn ssp_lines() -> impl Iterator<Item = String> {
    use std::io::BufRead;
    std::io::stdin().lock().lines().map_while(Result::ok).collect::<Vec<_>>().into_iter()
}
