// Does one kNN search block get slower as the store grows, with nothing else changed? tier_full7 (2 h 38 min, CE fine) had
// block times growing from 2 s to 70 s over 4.5M -> 5.0M keys; this times a 1024-query block (d = 256, tile 16384, the
// tier_eval shapes) at store sizes 0.5M .. 5M, then again at 5M after an idle minute, then on a fresh 0.5M store. Best and
// median of N rounds, so contention shows as a gap between them. Keys are Gaussian, not real hidden states: matmul time
// does not depend on the data, top-k insertion counts do, so read the ladder's shape, not its absolute values.
// Holds an exclusive GPU lease. Run: cargo run --release --example knn_scale_check
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::knn::KnnStore;
use scratchtape::nn::Rng;
use std::time::{Duration, Instant};

const D: usize = 256;
const TILE: usize = 16384;
const ROWS: usize = 1024;
const ROUNDS: usize = 5;

fn fill(store: &mut KnnStore, rng: &mut Rng, from: usize, to: usize) {
    for lo in (from..to).step_by(100_000) {
        let n = 100_000.min(to - lo);
        let keys: Vec<f32> = (0..n * D).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| ((lo + i) % 251) as u8).collect();
        store.add(&keys, &vals);
    }
    store.finish();
}

/// The online memory's write pattern: `window` keys at a time, no `finish()`. (With a `finish()` after each window, as
/// tier_eval used to do, every window became its own tile: 5M + 500k keys took 4.4 s per block instead of ~1 s.)
fn fill_small(store: &mut KnnStore, rng: &mut Rng, n: usize, window: usize) {
    for lo in (0..n).step_by(window) {
        let keys: Vec<f32> = (0..window * D).map(|_| rng.next_gaussian()).collect();
        store.add(&keys, &vec![(lo % 251) as u8; window]);
    }
}

fn time(store: &KnnStore, queries: &[f32], label: &str) {
    let mut t: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let t0 = Instant::now();
            let r = store.search(queries, ROWS);
            assert_eq!(r.len(), ROWS * scratchtape::gpu_step::knn::KMAX);
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{label:>28}: {} keys  best {:.0} ms  median {:.0} ms  worst {:.0} ms", store.len(), t[0], t[ROUNDS / 2], t[ROUNDS - 1]);
}

fn main() {
    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape knn_scale_check", Duration::from_secs(15 * 60));
    let mut rng = Rng::new(1);
    let queries: Vec<f32> = (0..ROWS * D).map(|_| rng.next_gaussian()).collect();
    let mut store = KnnStore::new(D, TILE);
    println!("f16 {}", store.is_f16());
    let mut n = 0;
    for target in [500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000] {
        fill(&mut store, &mut rng, n, target);
        n = target;
        time(&store, &queries, "ladder");
    }
    std::thread::sleep(Duration::from_secs(60));
    time(&store, &queries, "same 5M store after 60 s idle");
    // tier_eval's online writes: 32 keys (SEQ_LEN 64 - warm 32) per window; 500k keys.
    for added in [125_000, 250_000, 500_000] {
        let have = store.len() - 5_000_000;
        fill_small(&mut store, &mut rng, added - have, 32);
        time(&store, &queries, &format!("5M + {added} added 32 at a time"));
    }
    drop(store);
    let mut fresh = KnnStore::new(D, TILE);
    fill(&mut fresh, &mut rng, 0, 500_000);
    time(&fresh, &queries, "fresh 0.5M store");
}
