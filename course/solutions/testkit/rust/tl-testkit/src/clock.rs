//! The time seam: contracts take `&dyn Clock`, tests pass a
//! [`FakeClock`] whose `sleep` advances time at once and records the request.
use std::sync::Mutex;
use std::time::Duration;

pub trait Clock: Send + Sync {
    /// Monotonic time since an arbitrary origin.
    fn now(&self) -> Duration;
    fn sleep(&self, d: Duration);
}

pub struct RealClock {
    origin: std::time::Instant,
}

impl Default for RealClock {
    fn default() -> Self {
        RealClock {
            origin: std::time::Instant::now(),
        }
    }
}

impl Clock for RealClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d)
    }
}

#[derive(Default)]
pub struct FakeClock {
    t: Mutex<Duration>,
    sleeps: Mutex<Vec<Duration>>,
}

impl FakeClock {
    pub fn advance(&self, d: Duration) {
        *self.t.lock().unwrap() += d;
    }
    pub fn sleeps(&self) -> Vec<Duration> {
        self.sleeps.lock().unwrap().clone()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Duration {
        *self.t.lock().unwrap()
    }
    fn sleep(&self, d: Duration) {
        self.sleeps.lock().unwrap().push(d);
        self.advance(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_clock_moves_only_when_told() {
        let c = FakeClock::default();
        let t0 = c.now();
        c.sleep(Duration::from_millis(1500));
        c.advance(Duration::from_millis(500));
        assert_eq!(c.now() - t0, Duration::from_secs(2));
        assert_eq!(c.sleeps(), vec![Duration::from_millis(1500)]);
        let dynclock: &dyn Clock = &c;
        assert_eq!(dynclock.now(), Duration::from_secs(2));
    }
}
