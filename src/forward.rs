/* ------------------------------------------------------------------ */
/* Forward pass: per-token CPU and batched Metal paths               */
/* ------------------------------------------------------------------ */

use candle_core::{Result as CResult, Tensor, D};
use candle_nn::ops::softmax;
use crate::config::*;
use crate::metal::METAL_DEVICE;
use crate::model::{CandleModel, GPTModel, PosActs};
use crate::ops::{apply_dropout, linear_fwd, rmsnorm_fwd, softmax_fwd};
use crate::rng::Rng;

/// Per-token autoregressive forward pass with KV cache.
/// Returns (logits_per_position, activations_per_position).
/// Used during training so activations are available for backward.
///
/// `start_pos`: absolute sequence position of `tokens[0]`.  Pass 0 for a
/// full-sequence forward (training / prefill).  Pass the current context
/// length when decoding a single new token so that positional embeddings
/// and cache indices are correct.
pub fn forward(
    tokens: &[usize],
    model: &GPTModel,
    kv_cache: &mut Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    training: bool,
    mut rng: Option<&mut Rng>,
    start_pos: usize,
) -> (Vec<Vec<f32>>, Vec<PosActs>) {
    let seq_len = tokens.len();
    let mut all_logits = Vec::with_capacity(seq_len);
    let mut all_acts   = Vec::with_capacity(seq_len);

    for pos in 0..seq_len {
        let abs_pos = start_pos + pos;   // absolute position in the sequence
        let tok = tokens[pos];
        let mut act = PosActs::new();

        // Token + position embedding (use abs_pos for wpe)
        for i in 0..unsafe { N_EMBD } {
            act.x_embed[i] = model.wte[tok * unsafe { N_EMBD } + i] + model.wpe[abs_pos * unsafe { N_EMBD } + i];
        }

        let mut x = act.x_embed.clone();

        for li in 0..unsafe { N_LAYER } {
            act.x_in[li] = x.clone();

            // Attention pre-norm
            let mut xn = vec![0.0; unsafe { N_EMBD }];
            rmsnorm_fwd(&x, unsafe { N_EMBD }, &mut xn);
            act.xn_attn[li] = xn.clone();

            // Q, K, V projections
            let mut q = vec![0.0; unsafe { N_EMBD }];
            let mut k = vec![0.0; unsafe { N_EMBD }];
            let mut v = vec![0.0; unsafe { N_EMBD }];
            linear_fwd(&xn, &model.layers[li].wq, unsafe { N_EMBD }, unsafe { N_EMBD }, &mut q);
            linear_fwd(&xn, &model.layers[li].wk, unsafe { N_EMBD }, unsafe { N_EMBD }, &mut k);
            linear_fwd(&xn, &model.layers[li].wv, unsafe { N_EMBD }, unsafe { N_EMBD }, &mut v);

            act.q[li] = q.clone();
            act.k[li] = k.clone();
            act.v[li] = v.clone();

            // Append K,V to cache for abs_pos (only if not already cached)
            if kv_cache[li].len() <= abs_pos {
                kv_cache[li].push((k.clone(), v.clone()));
            }

            // Causal multi-head attention over all cached positions 0..=abs_pos
            let cache_len = abs_pos + 1;
            let mut attn_out = vec![0.0; unsafe { N_EMBD }];
            let scale = 1.0 / (unsafe { HEAD_DIM } as f32).sqrt();

            for h in 0..unsafe { N_HEAD } {
                let hs = h * unsafe { HEAD_DIM };
                let mut scores = vec![0.0; cache_len];
                for t in 0..cache_len {
                    let dot: f32 = (0..unsafe { HEAD_DIM })
                        .map(|j| q[hs + j] * kv_cache[li][t].0[hs + j])
                        .sum();
                    scores[t] = dot * scale;
                }
                let mut weights = vec![0.0; cache_len];
                softmax_fwd(&scores, cache_len, &mut weights, 1.0);
                for t in 0..cache_len {
                    for j in 0..HEAD_DIM {
                        attn_out[hs + j] += weights[t] * kv_cache[li][t].1[hs + j];
                    }
                }
            }

            act.attn_out[li] = attn_out.clone();

            // Output projection + optional dropout + residual
            let mut attn_proj = vec![0.0; unsafe { N_EMBD }];
            linear_fwd(&attn_out, &model.layers[li].wo, unsafe { N_EMBD }, unsafe { N_EMBD }, &mut attn_proj);
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut attn_proj, unsafe { DROPOUT_RATE }, r);
                }
            }
            for i in 0..unsafe { N_EMBD } { x[i] = attn_proj[i] + act.x_in[li][i]; }
            act.x_mid[li] = x.clone();

            // MLP pre-norm
            let mut xn_mlp = vec![0.0; unsafe { N_EMBD }];
            rmsnorm_fwd(&x, unsafe { N_EMBD }, &mut xn_mlp);
            act.xn_mlp[li] = xn_mlp.clone();

            // fc1 → squared ReLU → fc2 → optional dropout → residual
            let mut h1 = vec![0.0; unsafe { MLP_DIM }];
            linear_fwd(&xn_mlp, &model.layers[li].fc1, unsafe { MLP_DIM }, unsafe { N_EMBD }, &mut h1);
            act.mlp_pre[li] = h1.clone();

            let mut h2 = vec![0.0; unsafe { MLP_DIM }];
            for i in 0..unsafe { MLP_DIM } {
                h2[i] = if h1[i] > 0.0 { h1[i] * h1[i] } else { 0.0 };
            }
            act.mlp_post[li] = h2.clone();

            let mut mlp_out = vec![0.0; unsafe { N_EMBD }];
            linear_fwd(&h2, &model.layers[li].fc2, unsafe { N_EMBD }, unsafe { MLP_DIM }, &mut mlp_out);
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut mlp_out, unsafe { DROPOUT_RATE }, r);
                }
            }
            for i in 0..unsafe { N_EMBD } { x[i] = mlp_out[i] + act.x_mid[li][i]; }
        }

        act.x_out = x.clone();

        let mut logits = vec![0.0; model.vocab_size];
        linear_fwd(&x, &model.lm_head, model.vocab_size, unsafe { N_EMBD }, &mut logits);

        all_logits.push(logits);
        all_acts.push(act);
    }

    (all_logits, all_acts)
}

/// Metal-accelerated full-sequence forward pass (inference only, no activations).
/// Uses batched matmuls on the GPU for QKV / output / MLP projections.
/// Falls back to CPU if Metal is unavailable.
pub fn forward_metal_logits(tokens: &[usize], model: &GPTModel) -> Vec<Vec<f32>> {
    let device = match METAL_DEVICE.as_ref() {
        Some(d) => d,
        None    => return forward_metal_logits_cpu(tokens, model),
    };

    let seq_len = tokens.len();

    // Helper closure: [T, nin] * W^T [nout, nin] → flat [T*nout] on Metal
    let metal_mm = |x_data: &[f32], t: usize, nin: usize,
                    w_data: &[f32], nout: usize| -> Vec<f32> {
        let x_t = Tensor::from_slice(x_data, (t, nin), device).unwrap();
        let w_t = Tensor::from_slice(w_data, (nout, nin), device).unwrap();
        x_t.matmul(&w_t.t().unwrap()).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap()
    };

    // Build input embeddings [seq_len, N_EMBD]
    let mut x_flat = vec![0.0f32; seq_len * unsafe { N_EMBD }];
    for (pos, &tok) in tokens.iter().enumerate() {
        for i in 0..unsafe { N_EMBD } {
            x_flat[pos * unsafe { N_EMBD } + i] =
                model.wte[tok * unsafe { N_EMBD } + i] + model.wpe[pos * unsafe { N_EMBD } + i];
        }
    }

    for li in 0..unsafe { N_LAYER } {
        // Attention pre-norm (CPU — cheap elementwise)
        let mut xn_flat = vec![0.0f32; seq_len * unsafe { N_EMBD }];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_flat[pos * unsafe { N_EMBD }..(pos + 1) * unsafe { N_EMBD }],
                unsafe { N_EMBD },
                &mut xn_flat[pos * unsafe { N_EMBD }..(pos + 1) * unsafe { N_EMBD }],
            );
        }

        // Q, K, V on Metal
        let q_flat = metal_mm(&xn_flat, seq_len, unsafe { N_EMBD }, &model.layers[li].wq, unsafe { N_EMBD });
        let k_flat = metal_mm(&xn_flat, seq_len, unsafe { N_EMBD }, &model.layers[li].wk, unsafe { N_EMBD });
        let v_flat = metal_mm(&xn_flat, seq_len, unsafe { N_EMBD }, &model.layers[li].wv, unsafe { N_EMBD });

        // Causal attention (CPU — O(T²·d) but T=64 is small)
        let scale = 1.0 / (unsafe { HEAD_DIM } as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * unsafe { N_EMBD }];

        for h in 0..unsafe { N_HEAD } {
            let hs = h * unsafe { HEAD_DIM };
            for pos in 0..seq_len {
                let mut scores = vec![0.0f32; pos + 1];
                for t in 0..=pos {
                    let dot: f32 = (0..unsafe { HEAD_DIM })
                        .map(|j| q_flat[pos * unsafe { N_EMBD } + hs + j] * k_flat[t * unsafe { N_EMBD } + hs + j])
                        .sum();
                    scores[t] = dot * scale;
                }
                let mut weights = vec![0.0f32; pos + 1];
                softmax_fwd(&scores, pos + 1, &mut weights, 1.0);
                for t in 0..=pos {
                    for j in 0..unsafe { HEAD_DIM } {
                        attn_out[pos * unsafe { N_EMBD } + hs + j] +=
                            weights[t] * v_flat[t * unsafe { N_EMBD } + hs + j];
                    }
                }
            }
        }

        // Output projection on Metal
        let attn_proj = metal_mm(&attn_out, seq_len, unsafe { N_EMBD }, &model.layers[li].wo, unsafe { N_EMBD });

        // Residual
        let mut x_mid = vec![0.0f32; seq_len * unsafe { N_EMBD }];
        for i in 0..seq_len * unsafe { N_EMBD } { x_mid[i] = x_flat[i] + attn_proj[i]; }

        // MLP pre-norm (CPU)
        let mut xn_mlp = vec![0.0f32; seq_len * unsafe { N_EMBD }];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_mid[pos * unsafe { N_EMBD }..(pos + 1) * unsafe { N_EMBD }],
                unsafe { N_EMBD },
                &mut xn_mlp[pos * unsafe { N_EMBD }..(pos + 1) * unsafe { N_EMBD }],
            );
        }

        // fc1 on Metal
        let h1_flat = metal_mm(&xn_mlp, seq_len, unsafe { N_EMBD }, &model.layers[li].fc1, unsafe { MLP_DIM });

        // Squared ReLU (CPU — elementwise)
        let mut h2_flat = vec![0.0f32; seq_len * unsafe { MLP_DIM }];
        for i in 0..h2_flat.len() {
            let v = h1_flat[i];
            h2_flat[i] = if v > 0.0 { v * v } else { 0.0 };
        }

        // fc2 on Metal
        let mlp_out = metal_mm(&h2_flat, seq_len, unsafe { MLP_DIM }, &model.layers[li].fc2, unsafe { N_EMBD });

        // MLP residual
        for i in 0..seq_len * unsafe { N_EMBD } { x_flat[i] = x_mid[i] + mlp_out[i]; }
    }

    // LM head on Metal
    let logits_flat = metal_mm(&x_flat, seq_len, unsafe { N_EMBD }, &model.lm_head, model.vocab_size);

    logits_flat.chunks(model.vocab_size).map(|c| c.to_vec()).collect()
}

fn forward_metal_logits_cpu(tokens: &[usize], model: &GPTModel) -> Vec<Vec<f32>> {
    let mut kv = (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();
    let (logits, _) = forward(tokens, model, &mut kv, false, None, 0);
    logits
}

/// Fully-batched training forward pass via Candle autograd on Metal.
/// tokens:  [BATCH_SIZE, BLOCK_SIZE] u32
/// targets: [BATCH_SIZE, BLOCK_SIZE] u32
/// Returns a scalar cross-entropy loss Tensor.
pub fn forward_candle_train(
    tokens:  &Tensor,
    targets: &Tensor,
    model:   &CandleModel,
    training: bool,
) -> CResult<Tensor> {
    let device = &model.device;
    let (batch, seq_len) = tokens.dims2()?;

    // ── Embeddings ────────────────────────────────────────────────────
    // wte: [vocab, D], index per position → [B, T, D]
    let tok_flat = tokens.flatten_all()?;                             // [B*T]
    let tok_emb  = model.wte.as_tensor().index_select(&tok_flat, 0)? // [B*T, D]
        .reshape((batch, seq_len, unsafe { N_EMBD }))?;

    // wpe: [BLOCK_SIZE, D], slice first seq_len rows → [T, D] then broadcast
    let pos_emb = model.wpe.as_tensor().narrow(0, 0, seq_len)?       // [T, D]
        .unsqueeze(0)?;                                               // [1, T, D]

    let mut x = tok_emb.broadcast_add(&pos_emb)?;                    // [B, T, D]

    // ── Causal mask ───────────────────────────────────────────────────
    // Upper-triangular -inf, shape [1, 1, T, T] for broadcasting over heads
    let mask_data: Vec<f32> = (0..seq_len * seq_len)
        .map(|i| if (i % seq_len) <= (i / seq_len) { 0.0f32 } else { f32::NEG_INFINITY })
        .collect();
    let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?; // [1,1,T,T]

    // ── Transformer layers ────────────────────────────────────────────
    for li in 0..unsafe { N_LAYER } {
        let layer = &model.layers[li];

        // Attention pre-norm (RMSNorm: scale by 1/rms per token)
        let xn = {
            let sq = x.sqr()?;                                        // [B, T, D]
            let ms = sq.mean_keepdim(D::Minus1)?;                     // [B, T, 1]
            let scale = (ms + 1e-5f64)?.sqrt()?.recip()?;            // [B, T, 1]
            x.broadcast_mul(&scale)?                                  // [B, T, D]
        };

        // QKV projections: flatten [B,T,D]→[B*T,D], matmul [D,D]^T, reshape back
        let xn_2d = xn.reshape((batch * seq_len, unsafe { N_EMBD }))?;
        let q = xn_2d.matmul(&layer.wq.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { N_EMBD }))?;
        let k = xn_2d.matmul(&layer.wk.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { N_EMBD }))?;
        let v = xn_2d.matmul(&layer.wv.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { N_EMBD }))?;

        // Reshape to multi-head: [B, T, H, Dh] → [B, H, T, Dh]
        let q = q.reshape((batch, seq_len, unsafe { N_HEAD }, unsafe { HEAD_DIM }))?.transpose(1, 2)?.contiguous()?; // [B,H,T,Dh]
        let k = k.reshape((batch, seq_len, unsafe { N_HEAD }, unsafe { HEAD_DIM }))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, unsafe { N_HEAD }, unsafe { HEAD_DIM }))?.transpose(1, 2)?.contiguous()?;

        // Attention scores: [B, H, T, T]
        let scale_f = (unsafe { HEAD_DIM } as f64).sqrt().recip();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale_f, 0.0)?;
        let scores = scores.broadcast_add(&mask)?;                    // apply causal mask
        let attn_w = softmax(&scores, D::Minus1)?;                   // [B, H, T, T]

        // Weighted sum: [B, H, T, T] × [B, H, T, Dh] → [B, H, T, Dh]
        let attn_out = attn_w.matmul(&v)?;

        // Merge heads: [B, H, T, Dh] → [B, T, D]
        let attn_out = attn_out.transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, unsafe { N_EMBD }))?;

        // Output projection + residual: flatten → matmul → reshape → add
        let attn_2d = attn_out.reshape((batch * seq_len, unsafe { N_EMBD }))?;
        let proj = attn_2d.matmul(&layer.wo.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { N_EMBD }))?;
        let proj = if training { candle_nn::ops::dropout(&proj, unsafe { DROPOUT_RATE })? } else { proj };
        x = (x + proj)?;                                              // residual

        // MLP pre-norm
        let xn_mlp = {
            let sq = x.sqr()?;
            let ms = sq.mean_keepdim(D::Minus1)?;
            let scale = (ms + 1e-5f64)?.sqrt()?.recip()?;
            x.broadcast_mul(&scale)?
        };

        // fc1 → squared ReLU → fc2 + residual
        let xnm_2d = xn_mlp.reshape((batch * seq_len, unsafe { N_EMBD }))?;
        let h1  = xnm_2d.matmul(&layer.fc1.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { MLP_DIM }))?;
        let h1r = h1.relu()?;
        let h2  = h1r.mul(&h1r)?;                                     // squared ReLU [B, T, 4D]
        let h2_2d = h2.reshape((batch * seq_len, unsafe { MLP_DIM }))?;
        let mlp_out = h2_2d.matmul(&layer.fc2.as_tensor().t()?)?.reshape((batch, seq_len, unsafe { N_EMBD }))?;
        let mlp_out = if training { candle_nn::ops::dropout(&mlp_out, unsafe { DROPOUT_RATE })? } else { mlp_out };
        x = (x + mlp_out)?;
    }

    // ── LM head: [B, T, D] → [B*T, V] ───────────────────────────────
    let x_2d = x.reshape((batch * seq_len, unsafe { N_EMBD }))?;
    let logits_2d = x_2d.matmul(&model.lm_head.as_tensor().t()?)?;   // [B*T, V]

    // ── Cross-entropy loss ────────────────────────────────────────────
    let targets_flat = targets.flatten_all()?;                        // [B*T] u32
    candle_nn::loss::cross_entropy(&logits_2d, &targets_flat)
}

// ── Unit tests ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTModel;
    use crate::rng::Rng;

    /// Verify that prefill-then-single-token-decode produces identical logits
    /// to a monolithic full-sequence forward pass.
    #[test]
    fn kv_cache_single_token_matches_full_forward() {
        let mut rng = Rng::new(42);
        let vocab_size = 16;
        let model = GPTModel::new(vocab_size, &mut rng);

        // A short token sequence long enough to exercise multiple cache entries.
        let tokens: Vec<usize> = vec![3, 7, 1, 5, 2, 9];

        // ── Reference: full forward in one shot ──────────────────────────
        let mut kv_ref: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
            (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();
        let (full_logits, _) = forward(&tokens, &model, &mut kv_ref, false, None, 0);
        let ref_logits = full_logits.last().unwrap().clone();

        // ── Candidate: prefill [0..n-1], then single-token decode ────────
        let n = tokens.len();
        let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
            (0..unsafe { N_LAYER }).map(|_| Vec::new()).collect();
        // Prefill with all but the last token.
        forward(&tokens[..n - 1], &model, &mut kv_cache, false, None, 0);
        // Decode the last token.
        let (decode_logits, _) =
            forward(&tokens[n - 1..n], &model, &mut kv_cache, false, None, n - 1);
        let cand_logits = decode_logits.into_iter().next().unwrap();

        // ── Compare ──────────────────────────────────────────────────────
        assert_eq!(ref_logits.len(), cand_logits.len());
        for (i, (r, c)) in ref_logits.iter().zip(cand_logits.iter()).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit[{i}] mismatch: full={r:.6} vs cached={c:.6}"
            );
        }
    }
}
