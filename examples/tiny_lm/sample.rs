// Sample text from a checkpoint (the model alone: no lexicon, memory or dynamic evaluation).
//   sample <checkpoint> [corpus=fresh22] [offset=20000] [temp=0.8] [len=300] [seed=7] [seq=128] [d_model=256] [heads=8] [d_ff=512] [blocks=4]
// The prompt is the first `seq` bytes of the corpus's held-out text from `offset`; the output is printed next to the true
// continuation. temp=0 is greedy (it loops). Every byte reruns the whole window, so it is slow but simple.
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, read};
use scratchtape::nn::Rng;

#[path = "../common/mod.rs"]
mod common;
use common::{decode_bytes, split_corpus};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt: std::collections::HashMap<&str, &str> = args[1..].iter().map(|a| a.split_once('=').expect("key=value")).collect();
    let get = |k: &str, d: &str| opt.get(k).copied().unwrap_or(d).to_string();
    let num = |k: &str, d: &str| -> usize { get(k, d).parse().unwrap() };
    let (seq, len, offset) = (num("seq", "128"), num("len", "300"), num("offset", "20000"));
    let temp: f32 = get("temp", "0.8").parse().unwrap();
    let cfg = Config { vocab: 256, d: num("d_model", "256"), heads: num("heads", "8"), d_ff: num("d_ff", "512"), t: seq, n_blocks: num("blocks", "4"), softmax1: false };
    let flat: Vec<f32> = std::fs::read_to_string(&args[0]).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(flat.len(), cfg.len(), "{} does not match the model config", args[0]);
    let dev = DeviceParams::upload(&flat);
    let (_, held) = split_corpus(&get("corpus", "fresh22"));
    let mut rng = Rng::new(num("seed", "7") as u64);
    let mut ctx: Vec<usize> = held[offset..offset + seq].to_vec();
    let mut out = vec![];
    for _ in 0..len {
        let ids = ctx[ctx.len() - seq..].to_vec();
        let mut dt = DeviceTape::new(&dev);
        let (_, logits, _) = model_forward(&mut dt, &cfg, &ids, &ids, 1);
        let row = read(dt.value(logits))[(seq - 1) * 256..seq * 256].to_vec();
        let next = if temp == 0.0 {
            (0..256).max_by(|&x, &y| row[x].total_cmp(&row[y])).unwrap()
        } else {
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let p: Vec<f32> = row.iter().map(|x| ((x - m) / temp).exp()).collect();
            let mut u = rng.next_f32() * p.iter().sum::<f32>();
            p.iter().position(|&pj| { u -= pj; u < 0.0 }).unwrap_or(255)
        };
        ctx.push(next);
        out.push(next);
    }
    let show = |ids: &[usize]| decode_bytes(ids).replace('\n', " ");
    println!("PROMPT: {}\nMODEL:  {}\nACTUAL: {}", show(&held[offset..offset + seq]), show(&out), show(&held[offset + seq..offset + seq + len]));
}
