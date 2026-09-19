use scratchtape::nn::Rng;

/// Byte-level, not character-level: fixed universal vocab of 256 (the byte
/// value IS the token id, no lookup table or corpus scan needed to build a
/// vocabulary at all) - simpler than character-level tokenization, and the
/// more general "no tokenizer" answer (ByT5, MambaByte, Meta's Byte Latent
/// Transformer all operate this way). No Tokenizer struct/trait: this is
/// genuinely two one-liners, and per the established optimizer precedent
/// (SGD/Adam stayed separate concrete types, no shared trait, since nothing
/// needs runtime interchangeability), plug-and-play here just means picking
/// which function gets called - not polymorphism.
fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn decode_bytes(ids: &[usize]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Random contiguous window, target = the same window shifted by one
/// position (next-byte prediction). Random start via the existing Rng,
/// not sequential fixed chunks - standard practice, reuses infrastructure
/// already built rather than adding a new sampling scheme.
fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    let input = corpus[start..start + seq_len].to_vec();
    let target = corpus[start + 1..start + seq_len + 1].to_vec();
    (input, target)
}

/// Public domain (Shakespeare, Sonnet 18) - kept short given the naive
/// O(n^3) matmul this engine still uses; embedded as a string constant
/// rather than a file, matching every other demo's data (no file I/O
/// anywhere in this codebase yet).
const CORPUS: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

fn main() {
    let encoded = encode_bytes(CORPUS);
    println!("corpus length: {} bytes, vocab size: 256 (fixed)", encoded.len());

    // Round-trip check: encode then decode must reproduce the input exactly.
    let round_tripped = decode_bytes(&encoded);
    assert_eq!(round_tripped, CORPUS, "round-trip must be exact");
    println!("round-trip check: passed");

    let mut rng = Rng::new(1);
    let seq_len = 16;
    println!("\n3 sampled windows (input -> target, decoded):");
    for _ in 0..3 {
        let (input, target) = sample_window(&mut rng, &encoded, seq_len);
        println!("  {:?} -> {:?}", decode_bytes(&input), decode_bytes(&target));
    }
}
