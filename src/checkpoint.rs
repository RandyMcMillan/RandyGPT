/* ------------------------------------------------------------------ */
/* Checkpoint save / load                                            */
/* ------------------------------------------------------------------ */
//
// File format (little-endian):
//   [0..8]   magic      b"RGPT0001"
//   [8..12]  vocab_size u32
//   [12..16] iter       u32   (last completed iteration, 0-based)
//   [16..20] step       u32   (Adam step counter)
//   [20..24] best_loss  f32
//   [24..]   flat f32 arrays:
//              wte, wpe, lm_head,
//              per layer: wq, wk, wv, wo, fc1, fc2
//              m_wte, v_wte, m_wpe, v_wpe, m_lm_head, v_lm_head,
//              per layer: m_wq, v_wq, m_wk, v_wk, m_wv, v_wv,
//                         m_wo, v_wo, m_fc1, v_fc1, m_fc2, v_fc2

use std::fs::File;
use std::io::{Read, Write};
use crate::config::*;
use crate::model::{CandleModel, var_to_vec};
use crate::model::GPTModel;
use crate::optimizer::GpuAdamState;

// ── In-memory helpers ──────────────────────────────────────────────

fn write_f32s(buf: &mut Vec<u8>, s: &[f32]) {
    buf.reserve(s.len() * 4);
    for &v in s { buf.extend_from_slice(&v.to_le_bytes()); }
}

fn read_f32_slice(f: &mut File, n: usize) -> std::io::Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// ── Public API ─────────────────────────────────────────────────────

/// Serialize model + optimizer state to an in-memory byte buffer.
/// No disk I/O — call flush_checkpoint() to write to disk.
pub fn serialize_checkpoint(
    model: &GPTModel,
    iter: usize,
    step: usize,
    best_loss: f32,
) -> Vec<u8> {
    let n_params = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + unsafe { N_LAYER } * model.layers[0].wq.len() * 6;
    let mut buf: Vec<u8> = Vec::with_capacity(24 + n_params * 4 * 3);

    // Header
    buf.extend_from_slice(b"RGPT0001");
    buf.extend_from_slice(&(model.vocab_size as u32).to_le_bytes());
    buf.extend_from_slice(&(iter as u32).to_le_bytes());
    buf.extend_from_slice(&(step as u32).to_le_bytes());
    buf.extend_from_slice(&best_loss.to_le_bytes());

    // Weights
    write_f32s(&mut buf, &model.wte);
    write_f32s(&mut buf, &model.wpe);
    write_f32s(&mut buf, &model.lm_head);
    for li in 0..unsafe { N_LAYER } {
        let l = &model.layers[li];
        write_f32s(&mut buf, &l.wq);
        write_f32s(&mut buf, &l.wk);
        write_f32s(&mut buf, &l.wv);
        write_f32s(&mut buf, &l.wo);
        write_f32s(&mut buf, &l.fc1);
        write_f32s(&mut buf, &l.fc2);
    }

    // Adam moments
    write_f32s(&mut buf, &model.m_wte);    write_f32s(&mut buf, &model.v_wte);
    write_f32s(&mut buf, &model.m_wpe);    write_f32s(&mut buf, &model.v_wpe);
    write_f32s(&mut buf, &model.m_lm_head); write_f32s(&mut buf, &model.v_lm_head);
    for li in 0..unsafe { N_LAYER } {
        let l = &model.layers[li];
        write_f32s(&mut buf, &l.m_wq); write_f32s(&mut buf, &l.v_wq);
        write_f32s(&mut buf, &l.m_wk); write_f32s(&mut buf, &l.v_wk);
        write_f32s(&mut buf, &l.m_wv); write_f32s(&mut buf, &l.v_wv);
        write_f32s(&mut buf, &l.m_wo); write_f32s(&mut buf, &l.v_wo);
        write_f32s(&mut buf, &l.m_fc1); write_f32s(&mut buf, &l.v_fc1);
        write_f32s(&mut buf, &l.m_fc2); write_f32s(&mut buf, &l.v_fc2);
    }

    buf
}

/// Atomically flush a checkpoint buffer to disk (write to .tmp then rename).
pub fn flush_checkpoint(path: &str, buf: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", path);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(buf)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a checkpoint from disk into `model`.
/// Returns (iter_start, step, best_loss) — iter_start is the saved iter + 1
/// so training resumes *after* the last completed iteration.
pub fn load_checkpoint(
    path: &str,
    model: &mut GPTModel,
) -> std::io::Result<(usize, usize, f32)> {
    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0001" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Bad magic bytes in checkpoint {}", path),
        ));
    }

    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab_size {} != model vocab_size {}", ckpt_vocab, model.vocab_size),
        ));
    }

    model.wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..unsafe { N_LAYER } {
        let n_sq = unsafe { N_EMBD } * unsafe { N_EMBD };
        model.layers[li].wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].fc1 = read_f32_slice(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD })?;
        model.layers[li].fc2 = read_f32_slice(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM })?;
    }

    model.m_wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.v_wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.m_wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.v_wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.m_lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    model.v_lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..unsafe { N_LAYER } {
        let n_sq = unsafe { N_EMBD } * unsafe { N_EMBD };
        model.layers[li].m_wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_fc1 = read_f32_slice(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD })?;
        model.layers[li].v_fc1 = read_f32_slice(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD })?;
        model.layers[li].m_fc2 = read_f32_slice(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM })?;
        model.layers[li].v_fc2 = read_f32_slice(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM })?;
    }

    Ok((iter + 1, step, best_loss))
}

/// Load weights from any checkpoint format (v1/v2/v3) into a CPU GPTModel.
/// Ignores optimizer moments. Useful for --generate mode.
pub fn load_checkpoint_cpu(
    path: &str,
    model: &mut GPTModel,
) -> std::io::Result<(usize, usize, f32)> {
    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0001" && &magic != b"RGPT0002" && &magic != b"RGPT0003" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown checkpoint format in '{}': {:?}", path, magic),
        ));
    }

    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab {} != model vocab {}", ckpt_vocab, model.vocab_size),
        ));
    }

    // Weight layout is identical across v1/v2/v3
    model.wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..unsafe { N_LAYER } {
        let n_sq = unsafe { N_EMBD } * unsafe { N_EMBD };
        model.layers[li].wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].fc1 = read_f32_slice(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD })?;
        model.layers[li].fc2 = read_f32_slice(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM })?;
    }
    // Skip moments — not needed for inference

    Ok((iter + 1, step, best_loss))
}

// ── RGPT0002: CandleModel checkpoint (same binary layout, new magic) ──

/// Serialize a CandleModel to in-memory bytes (RGPT0002 format).
/// Kept for potential use; production code writes RGPT0003 via serialize_checkpoint_v3.
#[allow(dead_code)]
pub fn serialize_checkpoint_v2(
    model: &CandleModel,
    iter: usize,
    step: usize,
    best_loss: f32,
) -> Vec<u8> {
    let pull = |v: &candle_core::Var| -> Vec<f32> {
        v.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"RGPT0002");
    buf.extend_from_slice(&(model.vocab_size as u32).to_le_bytes());
    buf.extend_from_slice(&(iter as u32).to_le_bytes());
    buf.extend_from_slice(&(step as u32).to_le_bytes());
    buf.extend_from_slice(&best_loss.to_le_bytes());

    write_f32s(&mut buf, &pull(&model.wte));
    write_f32s(&mut buf, &pull(&model.wpe));
    write_f32s(&mut buf, &pull(&model.lm_head));
    for li in 0..unsafe { N_LAYER } {
        let l = &model.layers[li];
        write_f32s(&mut buf, &pull(&l.wq));
        write_f32s(&mut buf, &pull(&l.wk));
        write_f32s(&mut buf, &pull(&l.wv));
        write_f32s(&mut buf, &pull(&l.wo));
        write_f32s(&mut buf, &pull(&l.fc1));
        write_f32s(&mut buf, &pull(&l.fc2));
    }

    // RGPT0002 weights-only format — moments not stored (GpuAdamState handles them in v3)
    buf
}

/// Load an RGPT0002 checkpoint into a CandleModel.
pub fn load_checkpoint_v2(
    path: &str,
    model: &mut CandleModel,
) -> std::io::Result<(usize, usize, f32)> {
    use candle_core::{Tensor, Var};

    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0002" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected RGPT0002 magic in '{}', got {:?}", path, magic),
        ));
    }

    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab {} != model vocab {}", ckpt_vocab, model.vocab_size),
        ));
    }

    let device = &model.device.clone();
    let upload = |f: &mut File, n: usize, shape: (usize, usize)| -> std::io::Result<Var> {
        let data = read_f32_slice(f, n)?;
        Var::from_tensor(
            &Tensor::from_slice(&data, shape, device)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    };

    model.wte     = upload(&mut f, model.vocab_size * unsafe { N_EMBD }, (model.vocab_size, unsafe { N_EMBD }))?;
    model.wpe     = upload(&mut f, unsafe { BLOCK_SIZE } * unsafe { N_EMBD },       (unsafe { BLOCK_SIZE }, unsafe { N_EMBD }))?;
    model.lm_head = upload(&mut f, model.vocab_size * unsafe { N_EMBD }, (model.vocab_size, unsafe { N_EMBD }))?;

    for li in 0..unsafe { N_LAYER } {
        let n_sq = unsafe { N_EMBD } * unsafe { N_EMBD };
        model.layers[li].wq  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wk  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wv  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wo  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].fc1 = upload(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD }, (unsafe { MLP_DIM }, unsafe { N_EMBD }))?;
        model.layers[li].fc2 = upload(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM }, (unsafe { N_EMBD }, unsafe { MLP_DIM }))?;
    }

    // RGPT0002 was weights-only on disk — moments are zero-initialized by caller via GpuAdamState::new()

    Ok((iter + 1, step, best_loss))
}

// ── RGPT0003: CandleModel + GpuAdamState checkpoint ───────────────────────
//
// Same weights layout as RGPT0002, but Adam moments are now GPU Var tensors.
// step_t is restored from the header `step` field so bias correction is correct.
//
// Layout (little-endian):
//   [0..8)   "RGPT0003"
//   [8..12)  vocab_size (u32)
//   [12..16) iter       (u32)
//   [16..20) step       (u32)
//   [20..24) best_loss  (f32)
//   [24..)   Weights in all_vars() order:
//              wte, wpe, lm_head, [N_LAYER × wq wk wv wo fc1 fc2]
//            Adam first moments (m), same order
//            Adam second moments (v), same order

/// Serialize CandleModel + GpuAdamState to in-memory bytes (RGPT0003 format).
pub fn serialize_checkpoint_v3(
    model: &CandleModel,
    opt:   &GpuAdamState,
    iter: usize,
    step: usize,
    best_loss: f32,
) -> Vec<u8> {
    let pull_var = |v: &candle_core::Var| -> Vec<f32> {
        var_to_vec(v).unwrap_or_default()
    };
    let pull_mom = |v: &candle_core::Var| -> Vec<f32> {
        v.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"RGPT0003");
    buf.extend_from_slice(&(model.vocab_size as u32).to_le_bytes());
    buf.extend_from_slice(&(iter as u32).to_le_bytes());
    buf.extend_from_slice(&(step as u32).to_le_bytes());
    buf.extend_from_slice(&best_loss.to_le_bytes());

    // Weights in all_vars() order
    write_f32s(&mut buf, &pull_var(&model.wte));
    write_f32s(&mut buf, &pull_var(&model.wpe));
    write_f32s(&mut buf, &pull_var(&model.lm_head));
    for l in &model.layers {
        write_f32s(&mut buf, &pull_var(&l.wq));
        write_f32s(&mut buf, &pull_var(&l.wk));
        write_f32s(&mut buf, &pull_var(&l.wv));
        write_f32s(&mut buf, &pull_var(&l.wo));
        write_f32s(&mut buf, &pull_var(&l.fc1));
        write_f32s(&mut buf, &pull_var(&l.fc2));
    }
    // Adam first moments
    for m in &opt.m { write_f32s(&mut buf, &pull_mom(m)); }
    // Adam second moments
    for v in &opt.v { write_f32s(&mut buf, &pull_mom(v)); }

    buf
}

/// Load an RGPT0003 checkpoint into a CandleModel and GpuAdamState.
/// Returns (iter+1, step, best_loss).
pub fn load_checkpoint_v3(
    path:  &str,
    model: &mut CandleModel,
    opt:   &mut GpuAdamState,
) -> std::io::Result<(usize, usize, f32)> {
    use candle_core::{Tensor, Var};

    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0003" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected RGPT0003 magic in '{}', got {:?}", path, magic),
        ));
    }

    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab {} != model vocab {}", ckpt_vocab, model.vocab_size),
        ));
    }

    let device = &model.device.clone();
    let upload = |f: &mut File, n: usize, shape: (usize, usize)| -> std::io::Result<Var> {
        let data = read_f32_slice(f, n)?;
        Var::from_tensor(
            &Tensor::from_slice(&data, shape, device)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    };

    // Load weights
    model.wte     = upload(&mut f, model.vocab_size * unsafe { N_EMBD }, (model.vocab_size, unsafe { N_EMBD }))?;
    model.wpe     = upload(&mut f, unsafe { BLOCK_SIZE } * unsafe { N_EMBD },       (unsafe { BLOCK_SIZE }, unsafe { N_EMBD }))?;
    model.lm_head = upload(&mut f, model.vocab_size * unsafe { N_EMBD }, (model.vocab_size, unsafe { N_EMBD }))?;
    for li in 0..unsafe { N_LAYER } {
        let n_sq = unsafe { N_EMBD } * unsafe { N_EMBD };
        model.layers[li].wq  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wk  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wv  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].wo  = upload(&mut f, n_sq,             (unsafe { N_EMBD }, unsafe { N_EMBD }))?;
        model.layers[li].fc1 = upload(&mut f, unsafe { MLP_DIM } * unsafe { N_EMBD }, (unsafe { MLP_DIM }, unsafe { N_EMBD }))?;
        model.layers[li].fc2 = upload(&mut f, unsafe { N_EMBD } * unsafe { MLP_DIM }, (unsafe { N_EMBD }, unsafe { MLP_DIM }))?;
    }

    // Load Adam moments — shapes mirror all_vars() order
    let vars = model.all_vars();
    let n = vars.len();
    let mut m_new: Vec<Var> = Vec::with_capacity(n);
    let mut v_new: Vec<Var> = Vec::with_capacity(n);

    for var in &vars {
        let sz = var.as_tensor().elem_count();
        let data = read_f32_slice(&mut f, sz)?;
        let t = Tensor::from_slice(&data, var.shape(), device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        m_new.push(Var::from_tensor(&t)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?);
    }
    for var in &vars {
        let sz = var.as_tensor().elem_count();
        let data = read_f32_slice(&mut f, sz)?;
        let t = Tensor::from_slice(&data, var.shape(), device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        v_new.push(Var::from_tensor(&t)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?);
    }

    opt.m = m_new;
    opt.v = v_new;
    opt.step_t = step;

    Ok((iter + 1, step, best_loss))
}
