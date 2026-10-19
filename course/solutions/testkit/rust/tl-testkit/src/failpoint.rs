//! Named failpoints:
//!
//! ```text
//! TL_FAILPOINTS="engine/kv/after-push=crash;engine/decode=error(oom);x=3*sleep(20ms)"
//! ```
//!
//! Actions: `crash` (exit 137, what a SIGKILL leaves), `panic`, `error(msg)`
//! (`inject` returns `Err(msg)`), `sleep(d)` with `ms` or `s`, `off`. `N*`
//! fires only on the Nth evaluation.
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
enum Action {
    Crash,
    Panic,
    Error(String),
    Sleep(Duration),
    Off,
}

struct State {
    table: HashMap<String, (Action, u64)>,
    counts: HashMap<String, u64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim();
    if let Some(v) = s.strip_suffix("ms") {
        return v
            .parse::<u64>()
            .map(Duration::from_millis)
            .map_err(|e| e.to_string());
    }
    if let Some(v) = s.strip_suffix('s') {
        return v
            .parse::<f64>()
            .map(Duration::from_secs_f64)
            .map_err(|e| e.to_string());
    }
    Err(format!("sleep({s}): want a duration like 20ms or 2s"))
}

fn parse(spec: &str) -> Result<HashMap<String, (Action, u64)>, String> {
    let mut table = HashMap::new();
    for part in spec.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, act) = part
            .split_once('=')
            .ok_or_else(|| format!("failpoint {part:?}: want name=action"))?;
        let (nth, act) = match act.split_once('*') {
            Some((n, rest)) => (
                n.parse::<u64>()
                    .map_err(|_| format!("failpoint {part:?}: bad count"))?,
                rest,
            ),
            None => (0, act),
        };
        if act.contains('*')
            || (nth == 0 && act.len() != act.trim_start_matches(char::is_numeric).len())
        {
            return Err(format!("failpoint {part:?}: bad count"));
        }
        let (kind, arg) = match act.find('(') {
            Some(i) if act.ends_with(')') => (&act[..i], &act[i + 1..act.len() - 1]),
            _ => (act, ""),
        };
        let a = match kind {
            "crash" => Action::Crash,
            "panic" => Action::Panic,
            "error" => Action::Error(arg.to_string()),
            "sleep" => Action::Sleep(parse_duration(arg)?),
            "off" => Action::Off,
            _ => return Err(format!("failpoint {part:?}: unknown action {kind:?}")),
        };
        table.insert(name.trim().to_string(), (a, nth));
    }
    Ok(table)
}

/// Make `spec` the active set (tests); `inject` otherwise reads `TL_FAILPOINTS` once.
pub fn load(spec: &str) -> Result<(), String> {
    let table = parse(spec)?;
    *STATE.lock().unwrap() = Some(State {
        table,
        counts: HashMap::new(),
    });
    Ok(())
}

/// Evaluate a failpoint: `Ok(())` unless an `error` action fires.
pub fn inject(name: &str) -> Result<(), String> {
    let action = {
        let mut g = STATE.lock().unwrap();
        if g.is_none() {
            let spec = std::env::var("TL_FAILPOINTS").unwrap_or_default();
            let table = parse(&spec).unwrap_or_else(|e| panic!("TL_FAILPOINTS: {e}"));
            *g = Some(State {
                table,
                counts: HashMap::new(),
            });
        }
        let st = g.as_mut().unwrap();
        let Some((a, nth)) = st.table.get(name).cloned() else {
            return Ok(());
        };
        if a == Action::Off {
            return Ok(());
        }
        let c = st.counts.entry(name.to_string()).or_insert(0);
        *c += 1;
        if nth != 0 && *c != nth {
            return Ok(());
        }
        a
    };
    match action {
        Action::Crash => std::process::exit(137),
        Action::Panic => panic!("failpoint {name}"),
        Action::Sleep(d) => {
            std::thread::sleep(d);
            Ok(())
        }
        Action::Error(msg) => Err(format!("failpoint {name}: {msg}")),
        Action::Off => Ok(()),
    }
}

/// How many times a failpoint was evaluated while enabled.
pub fn count(name: &str) -> u64 {
    STATE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.counts.get(name).copied())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_and_counts() {
        load("a=error(disk full);b=2*error(second);c=off;s=sleep(1ms)").unwrap();
        assert_eq!(inject("a"), Err("failpoint a: disk full".to_string()));
        assert_eq!(inject("b"), Ok(()));
        assert!(inject("b").is_err());
        assert_eq!(inject("b"), Ok(()));
        assert_eq!(inject("c"), Ok(()));
        assert_eq!(inject("s"), Ok(()));
        assert_eq!(inject("nope"), Ok(()));
        assert_eq!(count("b"), 3);
        assert!(parse("x=boom").is_err());
        assert!(parse("x").is_err());
        assert!(parse("x=sleep(forever)").is_err());
    }
}
