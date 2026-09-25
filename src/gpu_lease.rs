//! GPU leases: taking turns on the shared eGPU (LONG_RUNS.md, "Take turns
//! on the eGPU"). Each GPU job keeps one JSON file,
//! `%LOCALAPPDATA%\gpu-leases\<pid>.json`:
//! `{"pid": 123, "kind": "exclusive", "purpose": "...", "until": <unix s>}`.
//! Benchmarks hold `Exclusive`. Long runs hold `Shared` and call
//! `pause_while_exclusive` at checkpoints. A file whose process has exited
//! or whose `until` has passed is stale and gets removed by whoever lists.
//! PowerShell sessions write the same format (snippet in LONG_RUNS.md).
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Shared,
    Exclusive,
}

/// A live lease read from the directory.
#[derive(Debug)]
pub struct Info {
    pub pid: u32,
    pub kind: Kind,
    pub purpose: String,
}

/// Our lease; dropping it deletes the file.
pub struct Lease {
    path: PathBuf,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA unset")).join("gpu-leases")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

/// Waits out other processes' exclusive leases, then takes one for this
/// process, valid for `ttl`. Prints the other live leases so a contended
/// run says so up front.
pub fn hold(kind: Kind, purpose: &str, ttl: Duration) -> Lease {
    pause_while_exclusive();
    let lease = hold_in(&dir(), kind, purpose, ttl);
    let others: Vec<_> = live().into_iter().filter(|l| l.pid != std::process::id()).collect();
    if others.is_empty() {
        eprintln!("gpu_lease: {kind:?} held; no other leases");
    } else {
        eprintln!("gpu_lease: {kind:?} held; others: {others:?}");
    }
    lease
}

fn hold_in(dir: &Path, kind: Kind, purpose: &str, ttl: Duration) -> Lease {
    std::fs::create_dir_all(dir).unwrap();
    let pid = std::process::id();
    let kind_s = if kind == Kind::Exclusive { "exclusive" } else { "shared" };
    let purpose = purpose.replace(['"', '\\'], "'");
    let text = format!("{{\"pid\": {pid}, \"kind\": \"{kind_s}\", \"purpose\": \"{purpose}\", \"until\": {}}}\n", now() + ttl.as_secs());
    let path = dir.join(format!("{pid}.json"));
    std::fs::write(&path, text).unwrap();
    Lease { path }
}

/// Every live lease; removes stale files along the way.
pub fn live() -> Vec<Info> {
    live_in(&dir())
}

fn live_in(dir: &Path) -> Vec<Info> {
    let Ok(entries) = std::fs::read_dir(dir) else { return vec![] };
    let mut out = vec![];
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        match parse(&text) {
            Some((info, until)) if until > now() && alive(info.pid) => out.push(info),
            _ => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    out
}

/// Blocks while another process holds a live exclusive lease. For long
/// runs, at checkpoint boundaries.
pub fn pause_while_exclusive() {
    pause_while_exclusive_in(&dir(), Duration::from_secs(10))
}

fn pause_while_exclusive_in(dir: &Path, poll: Duration) {
    let me = std::process::id();
    let mut said = false;
    loop {
        let held: Vec<_> = live_in(dir).into_iter().filter(|l| l.kind == Kind::Exclusive && l.pid != me).collect();
        if held.is_empty() {
            if said {
                eprintln!("gpu_lease: resuming");
            }
            return;
        }
        if !said {
            eprintln!("gpu_lease: pausing for {held:?}");
            said = true;
        }
        std::thread::sleep(poll);
    }
}

/// The value after `"key":`, unquoted. Tolerates PowerShell's
/// `ConvertTo-Json` spacing and a UTF-8 BOM.
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let rest = &text[text.find(&format!("\"{key}\""))? + key.len() + 2..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    if let Some(s) = rest.strip_prefix('"') {
        return Some(&s[..s.find('"')?]);
    }
    Some(rest[..rest.find([',', '}', '\n', '\r']).unwrap_or(rest.len())].trim())
}

fn parse(text: &str) -> Option<(Info, u64)> {
    let pid = field(text, "pid")?.parse().ok()?;
    let kind = match field(text, "kind")? {
        "exclusive" => Kind::Exclusive,
        "shared" => Kind::Shared,
        _ => return None,
    };
    let until = field(text, "until")?.parse().ok()?;
    Some((Info { pid, kind, purpose: field(text, "purpose").unwrap_or("").to_string() }, until))
}

#[cfg(windows)]
fn alive(pid: u32) -> bool {
    type Handle = *mut std::ffi::c_void;
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn GetExitCodeProcess(h: Handle, code: *mut u32) -> i32;
        fn CloseHandle(h: Handle) -> i32;
    }
    const QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    // Safety: plain Win32 calls; the handle is closed before returning.
    unsafe {
        let h = OpenProcess(QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut code = 0;
        let ok = GetExitCodeProcess(h, &mut code) != 0 && code == STILL_ACTIVE;
        CloseHandle(h);
        ok
    }
}

#[cfg(not(windows))]
fn alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file protocol: our lease is listed, PowerShell-formatted files
    /// parse, dead and expired leases are removed, dropping deletes ours,
    /// and our own exclusive lease never pauses us.
    #[test]
    fn leases_list_expire_and_release() {
        let dir = std::env::temp_dir().join(format!("gpu-leases-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let lease = hold_in(&dir, Kind::Exclusive, "test \"quoted\"", Duration::from_secs(60));
        let me = std::process::id();
        // As PowerShell's ConvertTo-Json + Out-File -Encoding utf8 writes it.
        let ps = format!("\u{feff}{{\n    \"until\":  {},\n    \"purpose\":  \"ps\",\n    \"kind\":  \"shared\",\n    \"pid\":  {me}\n}}\n", now() + 60);
        std::fs::write(dir.join("ps.json"), ps).unwrap();
        std::fs::write(dir.join("dead.json"), format!("{{\"pid\": 4294967294, \"kind\": \"exclusive\", \"until\": {}}}", now() + 60)).unwrap();
        std::fs::write(dir.join("old.json"), format!("{{\"pid\": {me}, \"kind\": \"exclusive\", \"until\": {}}}", now() - 1)).unwrap();

        let mut got: Vec<_> = live_in(&dir).into_iter().map(|l| (l.kind, l.purpose)).collect();
        got.sort_by_key(|g| g.1.clone());
        assert_eq!(got, vec![(Kind::Shared, "ps".to_string()), (Kind::Exclusive, "test 'quoted'".to_string())]);
        assert!(!dir.join("dead.json").exists() && !dir.join("old.json").exists());

        pause_while_exclusive_in(&dir, Duration::from_millis(1));
        drop(lease);
        assert!(!dir.join(format!("{me}.json")).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
