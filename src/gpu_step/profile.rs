//! Launch-site profilers for the device step: where host time goes
//! (queueing a step) and where GPU time goes (kernels). Both are off unless
//! started; `count_launch` in the parent module feeds them.
use super::client;
use std::collections::HashMap;
use std::panic::Location;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Host time per launch site, for finding where queueing a step goes: the
/// time from one launch's `count_launch` to the next is charged to the
/// first (its launch call, plus the host work before the next launch). Off
/// unless `host_profile_start`. Includes building the site key, ~1 µs.
static HOST: Mutex<Option<HostProfile>> = Mutex::new(None);

#[derive(Default)]
struct HostProfile {
    open: Option<(String, Instant)>,
    sites: HashMap<String, (usize, f64)>,
}

impl HostProfile {
    fn mark(&mut self, next: Option<String>) {
        let now = Instant::now();
        if let Some((site, t)) = self.open.take() {
            let e = self.sites.entry(site).or_default();
            e.0 += 1;
            e.1 += (now - t).as_secs_f64();
        }
        self.open = next.map(|n| (n, now));
    }
}

/// Per-launch-site GPU time from device timestamps, for finding slow
/// kernels. Each launch runs in its own profile window (no host sync; the
/// window only flushes the queue and brackets its compute pass with
/// timestamp writes), charged to the line that launched it. Off unless
/// `profile_start` is called. A sync-per-launch profiler was tried first
/// and couldn't rank kernels: its ~0.5 ms round trip swamped them.
static PROFILE: OnceLock<Mutex<Profile>> = OnceLock::new();

#[derive(Default)]
struct Profile {
    open: Option<(String, cubecl_runtime::client::ProfileWindow)>,
    done: Vec<(String, cubecl::profile::ProfileDuration)>,
}

impl Profile {
    fn mark(&mut self, next: Option<String>) {
        if let Some((site, w)) = self.open.take() {
            self.done.push((site, client().profile_end(w).unwrap()));
        }
        self.open = next.map(|n| (n, client().profile_start().unwrap()));
    }
}

/// Whether either profiler is running.
pub(super) fn active() -> bool {
    PROFILE.get().is_some() || HOST.lock().unwrap().is_some()
}

/// A launch at `at` (with `tag` appended to its site key): closes the open
/// window of each running profiler and opens the next.
pub(super) fn mark_launch(at: &Location, tag: String) {
    let site = format!("{}:{} {}", at.file(), at.line(), tag);
    if let Some(h) = HOST.lock().unwrap().as_mut() {
        h.mark(Some(site.clone()));
    }
    if let Some(p) = PROFILE.get() {
        p.lock().unwrap().mark(Some(site));
    }
}

/// (site, launches, total s) per site, most expensive first.
fn ranked(sites: HashMap<String, (usize, f64)>) -> Vec<(String, usize, f64)> {
    let mut v: Vec<_> = sites.into_iter().map(|(k, (n, t))| (k, n, t)).collect();
    v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    v
}

/// Starts charging host time to launch sites (see `HOST`).
pub fn host_profile_start() {
    *HOST.lock().unwrap() = Some(HostProfile::default());
}

/// Charges the open site up to now and stops charging until the next
/// launch; call it before a blocking readback so the wait isn't charged.
pub fn host_profile_cut() {
    if let Some(h) = HOST.lock().unwrap().as_mut() {
        h.mark(None);
    }
}

/// (site, launches, total host s), most expensive first; stops the
/// host profile.
pub fn host_profile_take() -> Vec<(String, usize, f64)> {
    let mut h = HOST.lock().unwrap().take().expect("host_profile_start first");
    h.mark(None);
    ranked(h.sites)
}

/// Starts charging GPU time to launch sites (see `PROFILE`). Panics if
/// the device can't report timestamps.
pub fn profile_start() {
    let method = client().properties().timing_method;
    assert_eq!(method, cubecl::profile::TimingMethod::Device, "no device timestamps");
    PROFILE.get_or_init(Default::default);
}

/// Closes the open window and returns (site, launches, total GPU s),
/// most expensive first; clears the tally. Windows that carried no
/// measurement are counted as launches but add no time.
pub fn profile_take() -> Vec<(String, usize, f64)> {
    let mut p = PROFILE.get().expect("profile_start first").lock().unwrap();
    p.mark(None);
    let mut sites: HashMap<String, (usize, f64)> = HashMap::new();
    for (site, d) in p.done.drain(..) {
        let e = sites.entry(site).or_default();
        e.0 += 1;
        if let Some(t) = pollster::block_on(d.resolve()) {
            e.1 += t.duration().as_secs_f64();
        }
    }
    ranked(sites)
}
