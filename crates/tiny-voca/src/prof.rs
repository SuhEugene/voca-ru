//! Optional stage timers: set `TINY_VOCA_PROFILE=1` and call `report()`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static TABLE: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static _UNUSED: AtomicU64 = AtomicU64::new(0);

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub struct Scope(Option<(&'static str, Instant)>);

#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope(ENABLED.load(Ordering::Relaxed).then(|| (name, Instant::now())))
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((name, started)) = self.0.take() {
            let nanos = started.elapsed().as_nanos() as u64;
            let mut table = TABLE.lock().unwrap();
            match table.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 += nanos,
                None => table.push((name, nanos)),
            }
        }
    }
}

pub fn report(runs: usize) {
    let table = TABLE.lock().unwrap();
    for (name, nanos) in table.iter() {
        eprintln!("{name:<24} {:9.3} ms", *nanos as f64 / 1e6 / runs as f64);
    }
}
