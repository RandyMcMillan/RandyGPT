/* ------------------------------------------------------------------ */
/* Tokenizer: char-level (default) or BPE (--bpe N flag)              */
/* ------------------------------------------------------------------ */
//
// Public interface is identical for both modes:
//   Tokenizer::from_text(text)              → char-level
//   Tokenizer::from_text_bpe(text, target)  → train BPE from scratch
//   Tokenizer::load_bpe(path)               → load saved BPE vocab
//   tokenizer.encode(text)  → Vec<usize>
//   tokenizer.decode(tokens) → String
//   tokenizer.vocab_size     → usize
//
// BPE vocab is saved/loaded as JSON (serde_json).
// Char-level checkpoints are NOT compatible with BPE checkpoints —
// the checkpoint vocab_size mismatch check in checkpoint.rs catches this.

use std::collections::HashMap;
use std::io;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ── Shared public struct ───────────────────────────────────────────────────

pub struct Tokenizer {
    pub vocab_size: usize,
    pub eos_id:     usize,
    #[allow(dead_code)]
    pub bos_id:     usize,
    mode:           TokenizerMode,
}

enum TokenizerMode {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

// ── Public API ─────────────────────────────────────────────────────────────

impl Tokenizer {
    // ── Char-level constructor (existing behaviour, unchanged) ──────────

    pub fn from_text(text: &str) -> Self {
        let ct = CharTokenizer::from_text(text);
        let vocab_size = ct.idx_to_char.len();
        let eos_id = ct.eos_id;
        let bos_id = ct.bos_id;
        Self { vocab_size, eos_id, bos_id, mode: TokenizerMode::Char(ct) }
    }

    // ── BPE constructor — train from scratch ────────────────────────────

    pub fn from_text_bpe(text: &str, target_vocab: usize) -> Self {
        let bt = BpeTokenizer::train(text, target_vocab);
        let vocab_size = bt.vocab.len();
        let eos_id     = bt.eos_id;
        let bos_id     = bt.bos_id;
        Self { vocab_size, eos_id, bos_id, mode: TokenizerMode::Bpe(bt) }
    }

    // ── BPE constructor — load from vocab.json ──────────────────────────

    pub fn load_bpe(path: &str) -> io::Result<Self> {
        let bt = BpeTokenizer::load(path)?;
        let vocab_size = bt.vocab.len();
        let eos_id     = bt.eos_id;
        let bos_id     = bt.bos_id;
        Ok(Self { vocab_size, eos_id, bos_id, mode: TokenizerMode::Bpe(bt) })
    }

    // ── Save BPE vocab to file (no-op for char-level) ───────────────────

    pub fn save_bpe(&self, path: &str) -> io::Result<()> {
        match &self.mode {
            TokenizerMode::Bpe(bt) => bt.save(path),
            TokenizerMode::Char(_) => Ok(()), // char-level has no persistent vocab
        }
    }

    pub fn _is_bpe(&self) -> bool {
        matches!(self.mode, TokenizerMode::Bpe(_))
    }

    // ── encode / decode ─────────────────────────────────────────────────

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match &self.mode {
            TokenizerMode::Char(ct) => ct.encode(text),
            TokenizerMode::Bpe(bt)  => bt.encode(text),
        }
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        match &self.mode {
            TokenizerMode::Char(ct) => ct.decode(tokens),
            TokenizerMode::Bpe(bt)  => bt.decode(tokens),
        }
    }

    // ── Convenience: first N vocab entries as display strings ───────────
    // Used in main.rs startup print.

    pub fn _sample_tokens(&self, n: usize) -> Vec<String> {
        let cap = n.min(self.vocab_size);
        match &self.mode {
            TokenizerMode::Char(ct) => {
                ct.idx_to_char[..cap].iter().map(|c| format!("{:?}", c)).collect()
            }
            TokenizerMode::Bpe(bt) => {
                bt.vocab[..cap].iter()
                    .map(|s| format!("{:?}", s))
                    .collect()
            }
        }
    }
}

// ── Char-level tokenizer ───────────────────────────────────────────────────

struct CharTokenizer {
    char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
    pub bos_id: usize,
    pub eos_id: usize,
}

impl CharTokenizer {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut idx_to_char = vec!['<']; // BOS = 0
        idx_to_char.push('>');           // EOS = 1
        idx_to_char.extend(chars);

        let char_to_idx: HashMap<char, usize> = idx_to_char
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        Self { char_to_idx, idx_to_char, bos_id: 0, eos_id: 1 }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }
}

// ── BPE tokenizer ──────────────────────────────────────────────────────────

// Serialisable vocab file format
#[derive(Serialize, Deserialize)]
struct BpeVocabFile {
    vocab:  Vec<String>,          // token_id → token_string
    merges: Vec<(String, String)>, // ordered merge rules (left, right)
}

struct BpeTokenizer {
    vocab:             Vec<String>,                    // token_id → string
    token_to_id:       HashMap<String, usize>,         // string → token_id
    merges:            Vec<(String, String)>,           // ordered merge rules
    merge_map:         HashMap<(usize, usize), usize>, // (left_id, right_id) → merged_id
    char_to_id:        HashMap<char, usize>,            // char → token_id (no String alloc per char)
    reverse_merge_map: HashMap<usize, (usize, usize)>, // merged_id → (left_id, right_id)
    pub bos_id:        usize,
    pub eos_id:        usize,
}

// Free helper: update pair_counts[pair] by delta, push fresh heap entry if > 0.
fn adjust(
    pair_counts: &mut HashMap<(usize, usize), i64>,
    heap: &mut std::collections::BinaryHeap<(i64, usize, usize)>,
    pair: (usize, usize),
    delta: i64,
) {
    let e = pair_counts.entry(pair).or_insert(0);
    *e += delta;
    if *e > 0 {
        heap.push((*e, pair.0, pair.1));
    }
}

impl BpeTokenizer {
    // ── Training ──────────────────────────────────────────────────────
    //
    // Algorithm: incremental BPE with a max-heap for O(log n) best-pair lookup.
    //
    // pair_counts: HashMap tracking true current count for each pair.
    // heap: BinaryHeap of (count, pair) — may contain stale entries.
    //   When we pop the heap, we check against pair_counts to skip stale entries.
    //   This is the standard "lazy deletion" heap pattern.
    //
    // After each merge we do one linear pass to apply the merge and update
    // pair_counts for affected neighbours only — O(occurrences) not O(corpus).

    fn train(text: &str, target_vocab: usize) -> Self {
        use std::collections::BinaryHeap;
        use std::io::Write;

        // Step 1: initial char vocabulary
        let mut vocab: Vec<String> = vec![
            "<|bos|>".to_string(),
            "<|eos|>".to_string(),
        ];
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        for c in &chars { vocab.push(c.to_string()); }

        let mut token_to_id: HashMap<String, usize> =
            vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

        // Step 2: initial corpus as char token IDs
        let mut corpus: Vec<usize> = text.chars()
            .filter_map(|c| token_to_id.get(&c.to_string()).copied())
            .collect();

        // Step 3: one full scan to build initial pair counts
        let mut pair_counts: HashMap<(usize, usize), i64> = HashMap::new();
        for w in corpus.windows(2) {
            *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
        }

        // Seed the heap: (count, Reverse(pair_string), pair) for max-by-count,
        // tie-break by lexicographic order of the merged string (via Reverse for min).
        // We use (count, pair) — stale entries are skipped on pop via lazy deletion.
        // Heap entry: (count, left_id, right_id) — max-heap on count.
        let mut heap: BinaryHeap<(i64, usize, usize)> = pair_counts.iter()
            .filter(|(_, &v)| v > 0)
            .map(|(&(l, r), &v)| (v, l, r))
            .collect();

        let initial_vocab = vocab.len();
        let n_merges = target_vocab.saturating_sub(initial_vocab);
        let mut merges: Vec<(String, String)> = Vec::new();

        let report_every = (n_merges / 20).max(1);

        for merge_idx in 0..n_merges {
            if merge_idx % report_every == 0 {
                let pct = merge_idx * 100 / n_merges;
                print!("\rBPE training: {}% ({}/{} merges, vocab {})   ",
                    pct, merge_idx, n_merges, vocab.len());
                let _ = std::io::stdout().flush();
            }

            // Pop heap until we find a non-stale entry
            let (left_id, right_id) = loop {
                match heap.pop() {
                    None => break (0, 0), // signal: done
                    Some((cached_count, l, r)) => {
                        let true_count = *pair_counts.get(&(l, r)).unwrap_or(&0);
                        if true_count > 0 && cached_count == true_count {
                            break (l, r);
                        }
                        // stale entry — skip
                    }
                }
            };
            if left_id == 0 && right_id == 0 { break; }

            let new_token = format!("{}{}", &vocab[left_id], &vocab[right_id]);
            let new_id = vocab.len();
            merges.push((vocab[left_id].clone(), vocab[right_id].clone()));
            token_to_id.insert(new_token.clone(), new_id);
            vocab.push(new_token);

            // One linear pass: apply merge, update pair_counts for neighbours only
            let mut new_corpus: Vec<usize> = Vec::with_capacity(corpus.len());
            let mut j = 0;
            while j < corpus.len() {
                if j + 1 < corpus.len()
                    && corpus[j] == left_id
                    && corpus[j + 1] == right_id
                {
                    // Update neighbour pairs
                    if let Some(&prev) = new_corpus.last() {
                        adjust(&mut pair_counts, &mut heap, (prev, left_id),  -1);
                        adjust(&mut pair_counts, &mut heap, (prev, new_id),    1);
                    }
                    adjust(&mut pair_counts, &mut heap, (left_id, right_id), -1);
                    if j + 2 < corpus.len() {
                        let after = corpus[j + 2];
                        adjust(&mut pair_counts, &mut heap, (right_id, after), -1);
                        adjust(&mut pair_counts, &mut heap, (new_id,   after),  1);
                    }
                    new_corpus.push(new_id);
                    j += 2;
                } else {
                    new_corpus.push(corpus[j]);
                    j += 1;
                }
            }
            corpus = new_corpus;
        }

        println!("\rBPE training: 100% ({} merges, vocab {})          ", merges.len(), vocab.len());

        // Build lookup tables for fast encoding
        let merge_map         = Self::build_merge_map(&vocab, &merges, &token_to_id);
        let char_to_id        = Self::build_char_map(&token_to_id);
        let reverse_merge_map = Self::build_reverse_merge_map(&merge_map);

        let bos_id = token_to_id["<|bos|>"];
        let eos_id = token_to_id["<|eos|>"];

        Self { vocab, token_to_id, merges, merge_map, char_to_id, reverse_merge_map, bos_id, eos_id }
    }

    fn build_merge_map(
        vocab: &[String],
        merges: &[(String, String)],
        token_to_id: &HashMap<String, usize>,
    ) -> HashMap<(usize, usize), usize> {
        let mut map = HashMap::new();
        for (left, right) in merges {
            if let (Some(&l), Some(&r)) = (token_to_id.get(left), token_to_id.get(right)) {
                let merged = format!("{}{}", left, right);
                if let Some(&m) = token_to_id.get(&merged) {
                    map.entry((l, r)).or_insert(m);
                    let _ = vocab;
                }
            }
        }
        map
    }

    /// Single-char tokens only — avoids String allocation per character during encoding.
    fn build_char_map(token_to_id: &HashMap<String, usize>) -> HashMap<char, usize> {
        token_to_id.iter()
            .filter_map(|(s, &id)| {
                let mut it = s.chars();
                let c = it.next()?;
                if it.next().is_none() { Some((c, id)) } else { None }
            })
            .collect()
    }

    /// Inverts merge_map so the inner apply-merge loop uses two int comparisons
    /// instead of a HashMap lookup per token pair.
    fn build_reverse_merge_map(
        merge_map: &HashMap<(usize, usize), usize>,
    ) -> HashMap<usize, (usize, usize)> {
        merge_map.iter().map(|(&(l, r), &m)| (m, (l, r))).collect()
    }

    // ── Save / Load ───────────────────────────────────────────────────

    fn save(&self, path: &str) -> io::Result<()> {
        let file = BpeVocabFile {
            vocab:  self.vocab.clone(),
            merges: self.merges.clone(),
        };
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, json)
    }

    fn load(path: &str) -> io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let file: BpeVocabFile = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::Other,
                format!("Failed to parse {}: {}", path, e)))?;

        let token_to_id: HashMap<String, usize> = file.vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let merge_map         = Self::build_merge_map(&file.vocab, &file.merges, &token_to_id);
        let char_to_id        = Self::build_char_map(&token_to_id);
        let reverse_merge_map = Self::build_reverse_merge_map(&merge_map);

        let bos_id = *token_to_id.get("<|bos|>").unwrap_or(&0);
        let eos_id = *token_to_id.get("<|eos|>").unwrap_or(&1);

        Ok(Self {
            vocab:             file.vocab,
            merges:            file.merges,
            token_to_id,
            merge_map,
            char_to_id,
            reverse_merge_map,
            bos_id,
            eos_id,
        })
    }

    // ── Encode ────────────────────────────────────────────────────────
    //
    // Algorithm: one pass per merge priority level.
    // We apply merges in training order (by merged token ID, ascending).
    // Each pass over the token sequence applies all instances of the
    // current-priority merge in a single left-to-right scan — O(n) per pass,
    // O(n × unique_merge_levels) total, which is fast for typical BPE depths.
    //
    // For large corpora, the public encode() splits on newlines and encodes
    // chunks in parallel with rayon, then concatenates.

    fn encode_chunk(
        merge_map:         &HashMap<(usize, usize), usize>,
        reverse_merge_map: &HashMap<usize, (usize, usize)>,
        char_to_id:        &HashMap<char, usize>,
        text: &str,
    ) -> Vec<usize> {
        // Char-level init — char_to_id avoids a String heap alloc per character.
        let mut tokens: Vec<usize> = text.chars()
            .filter_map(|c| char_to_id.get(&c).copied())
            .collect();

        if tokens.len() < 2 { return tokens; }

        // Repeatedly find the highest-priority applicable merge and apply it.
        loop {
            let best_merge = tokens.windows(2)
                .filter_map(|w| merge_map.get(&(w[0], w[1])).copied())
                .min(); // min merged_id = earliest trained = highest priority

            let merged_id = match best_merge {
                None => break,
                Some(m) => m,
            };

            // Reverse lookup: get (left_id, right_id) in O(1) — inner loop then uses
            // two integer comparisons instead of a HashMap lookup per token pair.
            let (left_id, right_id) = match reverse_merge_map.get(&merged_id) {
                Some(&pair) => pair,
                None => break,
            };

            let mut out: Vec<usize> = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len()
                    && tokens[i] == left_id
                    && tokens[i + 1] == right_id
                {
                    out.push(merged_id);
                    i += 2;
                } else {
                    out.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = out;
        }
        tokens
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        // Split on newlines so chunks are independent (no merge crosses a newline boundary,
        // which is fine — newline is its own token and never part of a merge with the
        // surrounding words in a Shakespeare corpus).
        let lines: Vec<&str> = text.split('\n').collect();
        let nl_id = self.token_to_id.get("\n").copied().unwrap_or(usize::MAX);
        let merge_map = &self.merge_map;
        let _token_to_id = &self.token_to_id;

        let reverse_merge_map = &self.reverse_merge_map;
        let char_to_id        = &self.char_to_id;

        let mut result: Vec<usize> = lines.par_iter()
            .flat_map(|line| {
                let mut chunk = Self::encode_chunk(merge_map, reverse_merge_map, char_to_id, line);
                chunk.push(nl_id); // re-add the newline we split on
                chunk
            })
            .collect();

        // Remove the trailing newline added after the last line
        if result.last() == Some(&nl_id) {
            result.pop();
        }
        result
    }

    // ── Decode ────────────────────────────────────────────────────────

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter(|&&id| id != self.eos_id && id != self.bos_id)
            .filter_map(|&id| self.vocab.get(id))
            .cloned()
            .collect()
    }
}
