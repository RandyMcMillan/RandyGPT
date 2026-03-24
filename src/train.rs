/* ------------------------------------------------------------------ */
/* Training loop, loss estimation, and text generation               */
/* ------------------------------------------------------------------ */

use std::time::Instant;
use rayon::prelude::*;

use candle_core::Tensor;
use crate::checkpoint::{flush_checkpoint, serialize_checkpoint, serialize_checkpoint_v3};
use crate::config::*;
use crate::forward::{forward, forward_candle_train, forward_metal_logits};
use crate::model::{CandleModel, GPTModel, GradientBuffer};
use crate::ops::{
    clip_gradients, cross_entropy_loss, linear_bwd_dx_only, linear_bwd_dw_batched,
    softmax_bwd, softmax_fwd,
};
use crate::optimizer::{adam_step, /*get_learning_rate, */zero_grads, GpuAdamState};
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::metal::METAL_DEVICE;

/* ------------------------------------------------------------------ */
/* Estimate loss on a dataset (Metal-accelerated where available)    */
/* ------------------------------------------------------------------ */
pub fn estimate_loss(
    model: &GPTModel,
    data: &[usize],
    valid_starts: &[usize],
    eval_iters: usize,
    rng: &mut Rng,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for _ in 0..eval_iters {
        if data.len() <= BLOCK_SIZE + 1 { continue; }

        let start = if !valid_starts.is_empty() {
            valid_starts[rng.choice(valid_starts.len())]
        } else {
            rng.choice(data.len() - BLOCK_SIZE - 1)
        };
        let x = &data[start..start + BLOCK_SIZE];
        let y = &data[start + 1..start + BLOCK_SIZE + 1];

        let logits_seq = if METAL_DEVICE.is_some() {
            forward_metal_logits(x, model)
        } else {
            let mut kv = (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();
            let (logits, _) = forward(x, model, &mut kv, false, None, 0);
            logits
        };

        let mut probs = vec![0.0f32; model.vocab_size];
        for (logits, &target) in logits_seq.iter().zip(y.iter()) {
            softmax_fwd(logits, model.vocab_size, &mut probs, 1.0);
            total_loss += cross_entropy_loss(&probs, target);
            count += 1;
        }
    }

    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

/* ------------------------------------------------------------------ */
/* Text generation with top-p sampling                               */
/* ------------------------------------------------------------------ */
pub fn generate(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
) -> String {
    generate_inner(model, tokenizer, prompt, max_new_tokens, temperature, top_p, rng, false)
}

pub fn generate_cpu(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
) -> String {
    generate_inner(model, tokenizer, prompt, max_new_tokens, temperature, top_p, rng, false)
}

pub fn generate_cpu_streaming(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
) {
    generate_inner(model, tokenizer, prompt, max_new_tokens, temperature, top_p, rng, true);
}

fn generate_inner(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
    stream: bool,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    // Empty prompt → seed with BOS so forward() has at least one token to work with.
    if tokens.is_empty() {
        tokens.push(tokenizer.bos_id);
    }

    // Helper: sample the next token from a logits vector using top-p (nucleus) sampling.
    let sample = |logits: &[f32], probs: &mut Vec<f32>, rng: &mut Rng| -> usize {
        softmax_fwd(logits, model.vocab_size, probs, temperature);
        let mut sorted: Vec<usize> = (0..model.vocab_size).collect();
        sorted.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff = model.vocab_size;
        for (i, &idx) in sorted.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p { cutoff = i + 1; break; }
        }
        let top_sum: f32 = sorted[..cutoff].iter().map(|&i| probs[i]).sum();
        let mut r = rng.uniform() as f32 * top_sum;
        let mut next_token = sorted[0];
        for i in 0..cutoff {
            let idx = sorted[i];
            r -= probs[idx];
            if r <= 0.0 { next_token = idx; break; }
        }
        next_token
    };

    let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
        (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();

    // Prefill: run the full prompt through forward() once to populate the KV cache.
    let (prefill_logits, _) = forward(&tokens, model, &mut kv_cache, false, None, 0);
    let mut last_logits = prefill_logits.last().unwrap().clone();
    let mut probs = vec![0.0f32; model.vocab_size];

    // Collect only the newly generated tokens so we never return the prompt.
    // Use generated.len() as the counter — tokens.len() shrinks on each slide
    // and can't be used to track progress once the context window is full.
    let mut generated: Vec<usize> = Vec::new();

    while generated.len() < max_new_tokens {
        let next_token = sample(&last_logits, &mut probs, rng);
        if next_token == tokenizer.eos_id { break; }
        if stream {
            use std::io::Write;
            print!("{}", tokenizer.decode(&[next_token]));
            let _ = std::io::stdout().flush();
        }
        generated.push(next_token);
        tokens.push(next_token);

        if tokens.len() > BLOCK_SIZE {
            // Slide the context window: keep last BLOCK_SIZE tokens.
            // Use Metal-accelerated forward for the heavy matmuls — KV cache
            // can't be reused after a slide (absolute position embeddings).
            tokens = tokens[tokens.len() - BLOCK_SIZE..].to_vec();
            let logits = forward_metal_logits(&tokens, model);
            last_logits = logits.into_iter().last().unwrap();
        } else {
            // Decode: process only the single new token at its absolute position.
            // The KV cache already holds entries for all prior positions, so this
            // is O(context_len) attention but only O(1) embedding + projection work.
            let abs_pos = tokens.len() - 1;
            let (decode_logits, _) =
                forward(&tokens[abs_pos..abs_pos + 1], model, &mut kv_cache, false, None, abs_pos);
            last_logits = decode_logits.into_iter().next().unwrap();
        }
    }

    tokenizer.decode(&generated)
}

/* ------------------------------------------------------------------ */
/* Main training loop                                                 */
/* ------------------------------------------------------------------ */
pub fn train(
    model: &mut GPTModel,
    data: &[usize],
    val_data: &[usize],
    valid_starts: &[usize],
    val_valid_starts: &[usize],
    iterations: usize,
    rng: &mut Rng,
    iter_start: usize,
    step_start: usize,
    best_loss_start: f32,
    max_lr: f32,
    min_lr: f32,
    checkpoint_prefix: &str,
) {
    println!("=== Starting Training (Multi-Core with Rayon) ===");
    if iter_start > 0 { println!("Resuming from iteration {}", iter_start); }
    println!("Iterations: {} → {}", iter_start, iterations);
    println!("Batch size: {}", unsafe { BATCH_SIZE });
    println!("Learning rate: {} → {}", max_lr, min_lr);
    println!("Gradient clipping: {}", GRAD_CLIP );
    println!("Cores available: {}", rayon::current_num_threads());
    println!();

    let mut step      = step_start;
    let mut best_loss = best_loss_start;
    let mut best_iter = if iter_start > 0 { iter_start.saturating_sub(1) } else { 0 };

    // In-memory checkpoint buffers — flushed only at end or Ctrl-C
    let mut ckpt_buf:      Vec<u8> = Vec::new();
    let mut ckpt_best_buf: Vec<u8> = Vec::new();

    let train_start  = Instant::now();
    let mut iter_count = 0u64;       // iterations actually executed this session
    let mut total_iter_ms = 0u64;    // cumulative wall time for those iterations
    let mut best_val_loss  = f32::INFINITY;
    let mut patience_count = 0usize;
    let mut stop_early     = false;
    let mut current_max_lr = max_lr; // ReduceLROnPlateau: decreases on plateau
    let mut lr_reductions  = 0usize;

    for iter in iter_start..iterations {
        let iter_start_time = Instant::now();
        // Sample batch indices — from valid_starts if available (no cross-doc windows)
        let batch_starts: Vec<usize> = (0..unsafe { BATCH_SIZE })
            .filter_map(|_| {
                if !valid_starts.is_empty() {
                    Some(valid_starts[rng.choice(valid_starts.len())])
                } else if data.len() > BLOCK_SIZE + 1 {
                    Some(rng.choice(data.len() - BLOCK_SIZE - 1))
                } else {
                    None
                }
            })
            .collect();

        if batch_starts.is_empty() { continue; }

        let model_ref = &*model;

        // Parallel forward + backward over batch items
        let results: Vec<(GradientBuffer, f32)> = batch_starts
            .par_iter()
            .map(|&start_idx| {
                let x_vec: Vec<usize> = data[start_idx..start_idx + BLOCK_SIZE].to_vec();
                let y_vec: Vec<usize> = data[start_idx + 1..start_idx + BLOCK_SIZE + 1].to_vec();

                let mut thread_rng = Rng::new(start_idx as u64 + iter as u64);

                let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
                    (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();
                let (logits_seq, acts) =
                    forward(&x_vec, model_ref, &mut kv_cache, true, Some(&mut thread_rng), 0);

                let seq_len = logits_seq.len();
                let mut local_grads = GradientBuffer::new(model_ref.vocab_size);
                let mut total_loss  = 0.0f32;

                // ── Per-position d_x pass ─────────────────────────────────
                // Compute d_x at every position (sequential — each pos depends
                // on earlier positions via the causal mask).  Along the way,
                // collect the activation matrices needed for the batched d_w
                // SGEMM pass below.
                //
                // Flat buffers: [seq_len × dim], row-major.
                // "d_" prefix = gradient of loss w.r.t. that activation.
                let mut d_x_all    = vec![0.0f32; seq_len * unsafe { N_EMBD }];   // d_x per position
                let mut d_logits_mat = vec![0.0f32; seq_len * model_ref.vocab_size];

                // Per-layer activation matrices for batched d_w
                // [layer][pos × dim]
                let mut mat_x_out      = vec![0.0f32; seq_len * unsafe { N_EMBD }];
                let mut mat_attn_out   = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut mat_xn_attn    = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut mat_xn_mlp     = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut mat_mlp_post   = vec![vec![0.0f32; seq_len * unsafe { MLP_DIM }]; unsafe { N_LAYER }];

                // d_out matrices (filled during position loop, used in SGEMM pass)
                let mut d_lm_head_d   = vec![0.0f32; seq_len * model_ref.vocab_size];
                let mut d_wo_d        = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut d_wq_d        = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut d_wk_d        = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut d_wv_d        = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];
                let mut d_fc1_d       = vec![vec![0.0f32; seq_len * unsafe { MLP_DIM }]; unsafe { N_LAYER }];
                let mut d_fc2_d       = vec![vec![0.0f32; seq_len * unsafe { N_EMBD }]; unsafe { N_LAYER }];

                // Populate activation matrices from saved acts
                for pos in 0..seq_len {
                    mat_x_out[pos * unsafe { N_EMBD } .. (pos+1)*unsafe { N_EMBD }]
                        .copy_from_slice(&acts[pos].x_out);
                    for li in 0..unsafe { N_LAYER } {
                        mat_attn_out[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }]
                            .copy_from_slice(&acts[pos].attn_out[li]);
                        mat_xn_attn[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }]
                            .copy_from_slice(&acts[pos].xn_attn[li]);
                        mat_xn_mlp[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }]
                            .copy_from_slice(&acts[pos].xn_mlp[li]);
                        mat_mlp_post[li][pos*unsafe { MLP_DIM }..(pos+1)*unsafe { MLP_DIM }]
                            .copy_from_slice(&acts[pos].mlp_post[li]);
                    }
                }

                let mut probs    = vec![0.0f32; model_ref.vocab_size];
                let mut d_x_out  = vec![0.0f32; unsafe { N_EMBD }];

                for pos in 0..seq_len {
                    softmax_fwd(&logits_seq[pos], model_ref.vocab_size, &mut probs, 1.0);
                    total_loss += cross_entropy_loss(&probs, y_vec[pos]);

                    // d_logits = probs - one_hot(target)
                    let dl = &mut d_logits_mat[pos * model_ref.vocab_size
                                              .. (pos+1) * model_ref.vocab_size];
                    dl.copy_from_slice(&probs);
                    dl[y_vec[pos]] -= 1.0;

                    // d_x_out via lm_head (d_w accumulated via SGEMM later)
                    d_x_out.fill(0.0);
                    {
                        let dl_slice = &d_logits_mat[pos * model_ref.vocab_size
                                                    .. (pos+1) * model_ref.vocab_size];
                        // d_x_out = lm_head^T · d_logits  (sgemv, no d_w here)
                                                linear_bwd_dx_only(
                                                    dl_slice, &model_ref.lm_head,
                                                    model_ref.vocab_size, unsafe { N_EMBD }, &mut d_x_out,
                                                );
                                                // store d_logits for SGEMM pass
                                                d_lm_head_d[pos * model_ref.vocab_size
                                                            .. (pos+1) * model_ref.vocab_size]
                                                    .copy_from_slice(dl_slice);
                                            }
                        
                                            let mut d_x = d_x_out.clone();
                        
                                            for li in (0..unsafe { N_LAYER }).rev() {
                                                // ----- MLP backward (d_x only) -----
                                                let mut d_h2 = vec![0.0f32; unsafe { MLP_DIM }];
                                                linear_bwd_dx_only(
                                                    &d_x, &model_ref.layers[li].fc2,
                                                    unsafe { N_EMBD }, unsafe { MLP_DIM }, &mut d_h2,
                                                );
                                                // store d_fc2_d for SGEMM
                                                d_fc2_d[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_x);
                        
                                                let mut d_h1 = vec![0.0f32; unsafe { MLP_DIM }];
                                                for i in 0..unsafe { MLP_DIM } {
                                                    if acts[pos].mlp_pre[li][i] > 0.0 {
                                                        d_h1[i] = d_h2[i] * 2.0 * acts[pos].mlp_pre[li][i];
                                                    }
                                                }
                                                // store d_fc1_d for SGEMM
                                                d_fc1_d[li][pos*unsafe { MLP_DIM }..(pos+1)*unsafe { MLP_DIM }].copy_from_slice(&d_h1);
                        
                                                let mut d_xn_mlp = vec![0.0f32; unsafe { N_EMBD }];
                                                linear_bwd_dx_only(
                                                    &d_h1, &model_ref.layers[li].fc1,
                                                    unsafe { MLP_DIM }, unsafe { N_EMBD }, &mut d_xn_mlp,
                                                );
                        
                                                let mut d_x_mid = vec![0.0f32; unsafe { N_EMBD }];
                                                for i in 0..unsafe { N_EMBD } { d_x_mid[i] = d_xn_mlp[i] + d_x[i]; }
                        
                                                // ----- Attention backward (d_x only) -----
                                                let mut d_attn_out = vec![0.0f32; unsafe { N_EMBD }];
                                                linear_bwd_dx_only(
                                                    &d_x_mid, &model_ref.layers[li].wo,
                                                    unsafe { N_EMBD }, unsafe { N_EMBD }, &mut d_attn_out,
                                                );
                                                // store d_wo_d for SGEMM
                                                d_wo_d[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_x_mid);
                        
                                                let mut d_q = vec![0.0f32; unsafe { N_EMBD }];
                                                let mut d_k = vec![0.0f32; unsafe { N_EMBD }];
                                                let mut d_v = vec![0.0f32; unsafe { N_EMBD }];
                                                let scale = 1.0 / (unsafe { HEAD_DIM } as f32).sqrt();
                        
                                                for h in 0..unsafe { N_HEAD } {
                                                    let hs = h * unsafe { HEAD_DIM };
                                                    let mut scores = vec![0.0f32; pos + 1];
                                                    for t in 0..=pos {
                                                        let dot: f32 = (0..unsafe { HEAD_DIM })
                                                            .map(|j| acts[pos].q[li][hs+j] * kv_cache[li][t].0[hs+j])
                                                            .sum();
                                                        scores[t] = dot * scale;
                                                    }
                                                    let mut attn_weights = vec![0.0f32; pos + 1];
                                                    softmax_fwd(&scores, pos + 1, &mut attn_weights, 1.0);
                        
                                                    let mut d_attn_weights = vec![0.0f32; pos + 1];
                                                    for t in 0..=pos {
                                                        for j in 0..unsafe { HEAD_DIM } {
                                                            d_attn_weights[t] +=
                                                                d_attn_out[hs+j] * kv_cache[li][t].1[hs+j];
                                                            if t == pos {
                                                                d_v[hs+j] += attn_weights[t] * d_attn_out[hs+j];
                                                            }
                                                        }
                                                    }
                                                    let mut d_scores = vec![0.0f32; pos + 1];
                                                    softmax_bwd(&attn_weights, &d_attn_weights, pos+1, &mut d_scores);
                                                    for t in 0..=pos {
                                                        for j in 0..unsafe { HEAD_DIM } {
                                                            d_q[hs+j] += d_scores[t] * scale * kv_cache[li][t].0[hs+j];
                                                            if t == pos {
                                                                d_k[hs+j] += d_scores[t] * scale * acts[pos].q[li][hs+j];
                                                            }
                                                        }
                                                    }
                                                }
                        
                                                // d_x through Q/K/V projections (d_w via SGEMM later)
                                                let mut d_xn_q = vec![0.0f32; unsafe { N_EMBD }];
                                                let mut d_xn_k = vec![0.0f32; unsafe { N_EMBD }];
                                                let mut d_xn_v = vec![0.0f32; unsafe { N_EMBD }];
                                                linear_bwd_dx_only(&d_q, &model_ref.layers[li].wq,
                                                    unsafe { N_EMBD }, unsafe { N_EMBD }, &mut d_xn_q);
                                                linear_bwd_dx_only(&d_k, &model_ref.layers[li].wk,
                                                    unsafe { N_EMBD }, unsafe { N_EMBD }, &mut d_xn_k);
                                                linear_bwd_dx_only(&d_v, &model_ref.layers[li].wv,
                                                    unsafe { N_EMBD }, unsafe { N_EMBD }, &mut d_xn_v);
                                                // store d_q/k/v for SGEMM
                                                d_wq_d[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_q);
                                                d_wk_d[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_k);
                                                d_wv_d[li][pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_v);
                        
                                                for i in 0..unsafe { N_EMBD } {
                                                    d_x[i] = d_xn_q[i] + d_xn_k[i] + d_xn_v[i] + d_x_mid[i];
                                                }
                                            }
                        
                                            // Embedding gradients
                                            for i in 0..unsafe { N_EMBD } {
                                                local_grads.d_wte[x_vec[pos] * unsafe { N_EMBD } + i] += d_x[i];
                                                local_grads.d_wpe[pos * unsafe { N_EMBD } + i]          += d_x[i];
                                            }
                                            d_x_all[pos*unsafe { N_EMBD }..(pos+1)*unsafe { N_EMBD }].copy_from_slice(&d_x);
                                        }
                        
                                        // ── Batched d_w pass via SGEMM ────────────────────────────
                                        // Each weight gradient = D^T · X  (one sgemm per matrix).
                                        // At seq_len=64 this replaces 64 sger calls with 1 sgemm,
                                        // letting AMX/SIMD process the full sequence in parallel.
                        
                                        // lm_head: d_w += d_logits^T · x_out
                                        linear_bwd_dw_batched(
                                            &d_lm_head_d, &mat_x_out,
                                            seq_len, model_ref.vocab_size, unsafe { N_EMBD },
                                            &mut local_grads.d_lm_head,
                                        );
                        
                                        for li in 0..unsafe { N_LAYER } {
                                            let sq   = unsafe { N_EMBD } * unsafe { N_EMBD };
                                            let fc1s = unsafe { MLP_DIM } * unsafe { N_EMBD };
                                            let fc2s = unsafe { N_EMBD } * unsafe { MLP_DIM };
                        
                                            // wo: d_w += d_wo_d^T · attn_out
                                            linear_bwd_dw_batched(
                                                &d_wo_d[li], &mat_attn_out[li],
                                                seq_len, unsafe { N_EMBD }, unsafe { N_EMBD },
                                                &mut local_grads.d_wo[li*sq..(li+1)*sq],
                                            );
                                            // wq, wk, wv: d_w += d_q/k/v^T · xn_attn
                                            linear_bwd_dw_batched(
                                                &d_wq_d[li], &mat_xn_attn[li],
                                                seq_len, unsafe { N_EMBD }, unsafe { N_EMBD },
                                                &mut local_grads.d_wq[li*sq..(li+1)*sq],
                                            );
                                            linear_bwd_dw_batched(
                                                &d_wk_d[li], &mat_xn_attn[li],
                                                seq_len, unsafe { N_EMBD }, unsafe { N_EMBD },
                                                &mut local_grads.d_wk[li*sq..(li+1)*sq],
                                            );
                                            linear_bwd_dw_batched(
                                                &d_wv_d[li], &mat_xn_attn[li],
                                                seq_len, unsafe { N_EMBD }, unsafe { N_EMBD },
                                                &mut local_grads.d_wv[li*sq..(li+1)*sq],
                                            );
                                            // fc1: d_w += d_fc1_d^T · xn_mlp
                                            linear_bwd_dw_batched(
                                                &d_fc1_d[li], &mat_xn_mlp[li],
                                                seq_len, unsafe { MLP_DIM }, unsafe { N_EMBD },
                                                &mut local_grads.d_fc1[li*fc1s..(li+1)*fc1s],
                                            );
                                            // fc2: d_w += d_fc2_d^T · mlp_post
                                            linear_bwd_dw_batched(
                                                &d_fc2_d[li], &mat_mlp_post[li],
                                                seq_len, unsafe { N_EMBD }, unsafe { MLP_DIM },
                                                &mut local_grads.d_fc2[li*fc2s..(li+1)*fc2s],
                                            );                }

                (local_grads, total_loss / seq_len as f32)
            })
            .collect();

        // Aggregate gradients sequentially
        zero_grads(model);
        let mut batch_loss = 0.0f32;

        for (grads, loss) in results {
            batch_loss += loss;
            model.d_wte.iter_mut().zip(grads.d_wte.iter()).for_each(|(a, b)| *a += b);
            model.d_wpe.iter_mut().zip(grads.d_wpe.iter()).for_each(|(a, b)| *a += b);
            model.d_lm_head.iter_mut().zip(grads.d_lm_head.iter()).for_each(|(a, b)| *a += b);
            for li in 0..unsafe { N_LAYER } {
                let sq   = unsafe { N_EMBD } * unsafe { N_EMBD };
                let fc1s = unsafe { MLP_DIM } * unsafe { N_EMBD };
                let fc2s = unsafe { N_EMBD } * unsafe { MLP_DIM };
                model.layers[li].d_wq.iter_mut().zip(grads.d_wq[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wk.iter_mut().zip(grads.d_wk[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wv.iter_mut().zip(grads.d_wv[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wo.iter_mut().zip(grads.d_wo[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_fc1.iter_mut().zip(grads.d_fc1[li*fc1s..(li+1)*fc1s].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_fc2.iter_mut().zip(grads.d_fc2[li*fc2s..(li+1)*fc2s].iter()).for_each(|(a,b)| *a+=b);
            }
        }

        batch_loss /= batch_starts.len() as f32;
        step += 1;

        let lr = {
            let decay_start = (iterations * 3) / 5;
            if iter < decay_start {
                current_max_lr
            } else {
                let progress = (iter - decay_start) as f32 / (iterations - decay_start) as f32;
                let cosine   = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
                min_lr + (current_max_lr - min_lr) * cosine
            }
        };

        // Gradient clipping
        clip_gradients(&mut model.d_wte, GRAD_CLIP );
        clip_gradients(&mut model.d_wpe, GRAD_CLIP );
        clip_gradients(&mut model.d_lm_head, GRAD_CLIP );
        for li in 0..unsafe { N_LAYER } {
            clip_gradients(&mut model.layers[li].d_wq,  GRAD_CLIP );
            clip_gradients(&mut model.layers[li].d_wk,  GRAD_CLIP );
            clip_gradients(&mut model.layers[li].d_wv,  GRAD_CLIP );
            clip_gradients(&mut model.layers[li].d_wo,  GRAD_CLIP );
            clip_gradients(&mut model.layers[li].d_fc1, GRAD_CLIP );
            clip_gradients(&mut model.layers[li].d_fc2, GRAD_CLIP );
        }

        // Adam optimizer update
        adam_step(&mut model.wte, &model.d_wte, &mut model.m_wte, &mut model.v_wte, step, lr);
        adam_step(&mut model.wpe, &model.d_wpe, &mut model.m_wpe, &mut model.v_wpe, step, lr);
        adam_step(&mut model.lm_head, &model.d_lm_head, &mut model.m_lm_head, &mut model.v_lm_head, step, lr);

        for li in 0..unsafe { N_LAYER } {
            // SAFETY: $w, $dw, $mw, $vw are distinct non-overlapping fields.
            let layer = &mut model.layers[li];
            macro_rules! layer_adam {
                ($w:ident, $dw:ident, $mw:ident, $vw:ident) => {{
                    let grads_ptr = layer.$dw.as_ptr();
                    let grads_len = layer.$dw.len();
                    let grads: &[f32] = unsafe { std::slice::from_raw_parts(grads_ptr, grads_len) };
                    adam_step(&mut layer.$w, grads, &mut layer.$mw, &mut layer.$vw, step, lr);
                }};
            }
            layer_adam!(wq, d_wq, m_wq, v_wq);
            layer_adam!(wk, d_wk, m_wk, v_wk);
            layer_adam!(wv, d_wv, m_wv, v_wv);
            layer_adam!(wo, d_wo, m_wo, v_wo);
            layer_adam!(fc1, d_fc1, m_fc1, v_fc1);
            layer_adam!(fc2, d_fc2, m_fc2, v_fc2);
        }

        if batch_loss < best_loss {
            best_loss = batch_loss;
            best_iter = iter;
        }

        iter_count    += 1;
        total_iter_ms += iter_start_time.elapsed().as_millis() as u64;

        // Log + snapshot checkpoint buffers
        let is_log_iter = iter % EVAL_INTERVAL == 0 || iter == iterations - 1;
        if is_log_iter {
            let elapsed = train_start.elapsed().as_secs_f32();
            let avg_ms  = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
            let remaining_iters = iterations.saturating_sub(iter + 1);
            let eta_s   = avg_ms * remaining_iters as f32 / 1000.0;
            let timing  = format!("{:.0}ms/iter | {:.0}s elapsed | ETA {:.0}s", avg_ms, elapsed, eta_s);

            if !val_data.is_empty() {
                let val_loss = estimate_loss(model, val_data, val_valid_starts, 50, rng);
                let val_ppl  = val_loss.exp();

                // ReduceLROnPlateau: reduce max_lr on patience exhaustion,
                // hard-stop only after MAX_LR_REDUCTIONS consecutive reductions.
                if EARLY_STOP_PATIENCE > 0 {
                    if val_loss < best_val_loss {
                        best_val_loss  = val_loss;
                        patience_count = 0;
                    } else {
                        patience_count += 1;
                        if patience_count >= EARLY_STOP_PATIENCE {
                            if lr_reductions < MAX_LR_REDUCTIONS {
                                current_max_lr = (current_max_lr * LR_REDUCTION_FACTOR ).max(min_lr);
                                lr_reductions += 1;
                                patience_count = 0;
                                println!(
                                    "  → Plateau: LR reduced to {:.2e} (reduction {}/{})",
                                    current_max_lr, lr_reductions, MAX_LR_REDUCTIONS )
                            } else {
                                stop_early = true;
                            }
                        }
                    }
                }

                let patience_str = if EARLY_STOP_PATIENCE > 0 {
                    format!(" | Pat: {}/{} LRx{}", patience_count, EARLY_STOP_PATIENCE , lr_reductions)
                } else {
                    String::new()
                };
                println!(
                    "Iter {:4} | Loss: {:.4} | Val: {:.4} (ppl {:.1}) | LR: {:.6} | Best: {:.4} @{} | {}{}",
                    iter, batch_loss, val_loss, val_ppl, lr, best_loss, best_iter, timing, patience_str
                );
            } else {
                println!(
                    "Iter {:4} | Loss: {:.4} | LR: {:.6} | Best: {:.4} @{} | {}",
                    iter, batch_loss, lr, best_loss, best_iter, timing
                );
            }
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            if best_iter == iter {
                ckpt_best_buf = ckpt_buf.clone();
                flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                    .unwrap_or_else(|e| eprintln!("Warning: could not save best checkpoint: {}", e));
            }

            // Early stopping break
            if stop_early {
                let elapsed = train_start.elapsed().as_secs_f32();
                let avg_ms  = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
                println!();
                println!("Early stopping: val loss hasn't improved for {} eval intervals ({} iters).",
                    EARLY_STOP_PATIENCE , EARLY_STOP_PATIENCE  * EVAL_INTERVAL );
                println!("Best val loss was {:.4} @{}. Saving checkpoint and stopping.", best_loss, best_iter);
                println!("Total time: {:.1}s | Avg: {:.0}ms/iter", elapsed, avg_ms);
                flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
                    .map(|_| println!("✓ Saved {}.bin (iter {})", checkpoint_prefix, iter))
                    .unwrap_or_else(|e| eprintln!("Warning: {}", e));
                if !ckpt_best_buf.is_empty() {
                    flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                        .map(|_| println!("✓ Saved {}_best.bin (best loss {:.4} @{})", checkpoint_prefix, best_loss, best_iter))
                        .unwrap_or_else(|e| eprintln!("Warning: {}", e));
                }
                return;
            }
        }

        // Ctrl-C: flush and exit
        if ctrlc_tiny::is_ctrlc_received() {
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            if best_iter == iter || ckpt_best_buf.is_empty() {
                ckpt_best_buf = ckpt_buf.clone();
            }
            let elapsed   = train_start.elapsed().as_secs_f32();
            let avg_ms    = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
            println!();
            println!("Interrupted at iteration {}. Saving checkpoint...", iter);
            println!("Elapsed: {:.1}s | Avg: {:.0}ms/iter", elapsed, avg_ms);
            flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
                .map(|_| println!("✓ Saved {}.bin (iter {})", checkpoint_prefix, iter))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                .map(|_| println!("✓ Saved {}_best.bin (best loss {:.4} @{})", checkpoint_prefix, best_loss, best_iter))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            std::process::exit(0);
        }
    }

    let total_elapsed = train_start.elapsed().as_secs_f32();
    let avg_ms = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };

    println!();
    println!("Training complete!");
    println!("Total time:  {:.1}s | Avg: {:.0}ms/iter ({} iters)", total_elapsed, avg_ms, iter_count);
    println!("Best loss: {:.4} at iteration {}", best_loss, best_iter);

    if !ckpt_buf.is_empty() {
        flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save checkpoint: {}", e));
    }
    if !ckpt_best_buf.is_empty() {
        flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save best checkpoint: {}", e));
    }
}

/* ------------------------------------------------------------------ */
/* Candle (Metal GPU) training loop                                   */
/* ------------------------------------------------------------------ */
pub fn train_candle(
    model: &mut CandleModel,
    opt: &mut GpuAdamState,
    data: &[usize],
    val_data: &[usize],
    valid_starts: &[usize],
    val_valid_starts: &[usize],
    iterations: usize,
    rng: &mut Rng,
    iter_start: usize,
    step_start: usize,
    best_loss_start: f32,
    max_lr: f32,
    min_lr: f32,
    checkpoint_prefix: &str,
) {
    println!("=== Starting Training (Metal GPU via Candle) ===");
    if iter_start > 0 { println!("Resuming from iteration {}", iter_start); }
    println!("Iterations: {} → {}", iter_start, iterations);
    println!("Batch size: {} × {} accum steps = {} effective", BATCH_SIZE , GRAD_ACCUM_STEPS , BATCH_SIZE * GRAD_ACCUM_STEPS );
    println!("Learning rate: {} → {}", max_lr, min_lr);
    println!();

    let device = model.device.clone();
    let mut step       = step_start;

    let mut ckpt_buf:      Vec<u8> = Vec::new();
    let mut ckpt_best_buf: Vec<u8> = Vec::new();

    let train_start    = Instant::now();
    let mut iter_count = 0u64;
    let mut total_iter_ms = 0u64;
    let mut best_val_loss   = best_loss_start; // tracks val loss for early stopping; seeded from checkpoint
    let mut patience_count  = 0usize;          // consecutive evals with no val improvement
    let mut stop_early      = false;
    let mut current_max_lr  = max_lr;          // ReduceLROnPlateau: decreases on plateau
    let mut lr_reductions   = 0usize;

    for iter in iter_start..iterations {
        let iter_start_time = Instant::now();

        // ── Gradient accumulation loop ────────────────────────────────
        // Run GRAD_ACCUM_STEPS micro-batches, sum their losses, then do
        // one backward + optimizer step. Effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS.
        let mut accum_loss: Option<Tensor> = None;
        let mut batch_loss_sum = 0.0f32;
        let mut accum_count = 0usize;

        for _ in 0..GRAD_ACCUM_STEPS {
            let mut tok_data: Vec<u32> = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE );
            let mut tgt_data: Vec<u32> = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE );
            for _ in 0..BATCH_SIZE {
                if data.len() <= BLOCK_SIZE + 1 { continue; }
                let start = if !valid_starts.is_empty() {
                    valid_starts[rng.choice(valid_starts.len())]
                } else {
                    rng.choice(data.len() - BLOCK_SIZE - 1)
                };
                for t in 0..BLOCK_SIZE {
                    tok_data.push(data[start + t] as u32);
                    tgt_data.push(data[start + t + 1] as u32);
                }
            }
            let actual_batch = tok_data.len() / BLOCK_SIZE;
            if actual_batch == 0 { continue; }

            let tokens  = Tensor::from_vec(tok_data, (actual_batch, BLOCK_SIZE), &device)
                .unwrap_or_else(|e| panic!("token tensor: {}", e));
            let targets = Tensor::from_vec(tgt_data, (actual_batch, BLOCK_SIZE), &device)
                .unwrap_or_else(|e| panic!("target tensor: {}", e));

            let loss = forward_candle_train(&tokens, &targets, model, true)
                .unwrap_or_else(|e| panic!("forward: {}", e));

            batch_loss_sum += loss.to_scalar::<f32>()
                .unwrap_or_else(|e| panic!("loss scalar: {}", e));
            accum_count += 1;

            // Accumulate into sum — backward on the mean across micro-batches
            accum_loss = Some(match accum_loss {
                None       => loss,
                Some(prev) => (prev + loss).unwrap_or_else(|e| panic!("loss add: {}", e)),
            });
        }

        if accum_count == 0 { continue; }
        let batch_loss = batch_loss_sum / accum_count as f32;

        // Divide accumulated loss by step count so gradients are mean-scaled
        let mean_loss = (accum_loss.unwrap() * (1.0 / accum_count as f64))
            .unwrap_or_else(|e| panic!("loss scale: {}", e));

        // ── Single backward + optimizer step ──────────────────────────
        let grads = mean_loss.backward()
            .unwrap_or_else(|e| panic!("backward: {}", e));

        step += 1;
        let lr = {
            let decay_start = (iterations * 3) / 5;
            if iter < decay_start {
                current_max_lr
            } else {
                let progress = (iter - decay_start) as f32 / (iterations - decay_start) as f32;
                let cosine   = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
                min_lr + (current_max_lr - min_lr) * cosine
            }
        };

        // ── Full GPU AdamW: clip + update all Vars on Metal, no CPU transfer ──
        let vars = model.all_vars();
        opt.step(&grads, &vars, lr)
            .unwrap_or_else(|e| panic!("adam step: {}", e));

        iter_count    += 1;
        total_iter_ms += iter_start_time.elapsed().as_millis() as u64;

        // ── Log + checkpoint ──────────────────────────────────────────
        let is_log_iter = iter % EVAL_INTERVAL == 0 || iter == iterations - 1;
        if is_log_iter {
            let elapsed = train_start.elapsed().as_secs_f32();
            let avg_ms  = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
            let remaining_iters = iterations.saturating_sub(iter + 1);
            let eta_s = avg_ms * remaining_iters as f32 / 1000.0;
            let timing = format!("{:.0}ms/iter | {:.0}s elapsed | ETA {:.0}s", avg_ms, elapsed, eta_s);

            let mut new_best = false;
            if !val_data.is_empty() {
                // Use CPU model for val loss (forward_metal_logits path unchanged)
                let cpu_model = model.to_gpt().unwrap_or_else(|e| panic!("to_gpt: {}", e));
                let val_loss = estimate_loss(&cpu_model, val_data, val_valid_starts, 50, rng);
                let val_ppl  = val_loss.exp();

                // ReduceLROnPlateau: reduce max_lr on patience exhaustion,
                // hard-stop only after MAX_LR_REDUCTIONS consecutive reductions.
                new_best = val_loss < best_val_loss;
                if EARLY_STOP_PATIENCE > 0 {
                    if new_best {
                        best_val_loss  = val_loss;
                        patience_count = 0;
                    } else {
                        patience_count += 1;
                        if patience_count >= EARLY_STOP_PATIENCE {
                            if lr_reductions < MAX_LR_REDUCTIONS {
                                current_max_lr = (current_max_lr * LR_REDUCTION_FACTOR ).max(min_lr);
                                lr_reductions += 1;
                                patience_count = 0;
                                println!(
                                    "  → Plateau: LR reduced to {:.2e} (reduction {}/{})",
                                    current_max_lr, lr_reductions, MAX_LR_REDUCTIONS )
                            } else {
                                stop_early = true;
                            }
                        }
                    }
                }

                let patience_str = if EARLY_STOP_PATIENCE > 0 {
                    format!(" | Pat: {}/{} LRx{}", patience_count, EARLY_STOP_PATIENCE , lr_reductions)
                } else {
                    String::new()
                };
                println!(
                    "Iter {:4} | Loss: {:.4} | Val: {:.4} (ppl {:.1}) | LR: {:.6} | Best val: {:.4} | {}{}",
                    iter, batch_loss, val_loss, val_ppl, lr, best_val_loss, timing, patience_str
                );
            } else {
                println!(
                    "Iter {:4} | Loss: {:.4} | LR: {:.6} | Best val: {:.4} | {}",
                    iter, batch_loss, lr, best_val_loss, timing
                );
            }

            ckpt_buf = serialize_checkpoint_v3(model, opt, iter, step, best_val_loss);
            // checkpoint_best tracks best VAL loss, not train loss
            if new_best || ckpt_best_buf.is_empty() {
                ckpt_best_buf = ckpt_buf.clone();
                if new_best {
                    flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                        .unwrap_or_else(|e| eprintln!("Warning: could not save best checkpoint: {}", e));
                }
            }

            // ── Early stopping ────────────────────────────────────────
            if stop_early {
                let elapsed = train_start.elapsed().as_secs_f32();
                let avg_ms  = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
                println!();
                println!("Early stopping: val loss hasn't improved for {} eval intervals ({} iters).",
                    EARLY_STOP_PATIENCE , EARLY_STOP_PATIENCE  * EVAL_INTERVAL );
                println!("Best val loss was {:.4}. Saving checkpoint and stopping.", best_val_loss);
                println!("Total time: {:.1}s | Avg: {:.0}ms/iter", elapsed, avg_ms);
                flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
                    .map(|_| println!("✓ Saved {}.bin (iter {})", checkpoint_prefix, iter))
                    .unwrap_or_else(|e| eprintln!("Warning: {}", e));
                if !ckpt_best_buf.is_empty() {
                    flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                        .map(|_| println!("✓ Saved {}_best.bin (best val loss {:.4})", checkpoint_prefix, best_val_loss))
                        .unwrap_or_else(|e| eprintln!("Warning: {}", e));
                }
                return;
            }
        }

        // ── Ctrl-C ────────────────────────────────────────────────────
        if ctrlc_tiny::is_ctrlc_received() {
            ckpt_buf = serialize_checkpoint_v3(model, opt, iter, step, best_val_loss);
            if ckpt_best_buf.is_empty() { ckpt_best_buf = ckpt_buf.clone(); }
            let elapsed = train_start.elapsed().as_secs_f32();
            let avg_ms  = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
            println!();
            println!("Interrupted at iteration {}. Saving checkpoint...", iter);
            println!("Elapsed: {:.1}s | Avg: {:.0}ms/iter", elapsed, avg_ms);
            flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
                .map(|_| println!("✓ Saved {}.bin (iter {})", checkpoint_prefix, iter))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
                .map(|_| println!("✓ Saved {}_best.bin (best val loss {:.4})", checkpoint_prefix, best_val_loss))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            std::process::exit(0);
        }
    }

    let total_elapsed = train_start.elapsed().as_secs_f32();
    let avg_ms = if iter_count > 0 { total_iter_ms as f32 / iter_count as f32 } else { 0.0 };
    println!();
    println!("Training complete! (Metal GPU)");
    println!("Total time:  {:.1}s | Avg: {:.0}ms/iter ({} iters)", total_elapsed, avg_ms, iter_count);
    println!("Best val loss: {:.4}", best_val_loss);

    if !ckpt_buf.is_empty() {
        flush_checkpoint(&format!("{}.bin", checkpoint_prefix), &ckpt_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save checkpoint: {}", e));
    }
    if !ckpt_best_buf.is_empty() {
        flush_checkpoint(&format!("{}_best.bin", checkpoint_prefix), &ckpt_best_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save best checkpoint: {}", e));
    }
}
