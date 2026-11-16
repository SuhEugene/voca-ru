//! Numeric kernels: matrix product, linear layers, normalization, activations.
//! Matrices are row-major `[rows, columns]` slices of `f32`.

use rayon::prelude::*;

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod blas {
    use std::os::raw::c_int;
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        pub fn cblas_sgemm(order: c_int, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: *const f32, lda: c_int,
                           b: *const f32, ldb: c_int, beta: f32, c: *mut f32, ldc: c_int);
    }
    pub const ROW_MAJOR: c_int = 101;
    pub const NO_TRANS: c_int = 111;
    pub const TRANS: c_int = 112;
}

/// Name of the matrix-product backend compiled in.
pub fn backend() -> &'static str {
    if cfg!(all(target_os = "macos", feature = "accelerate")) { "accelerate" } else { "matrixmultiply" }
}

/// `C = alpha * A * B + beta * C` for strided matrices: A is `m x k`, B is
/// `k x n`, C is `m x n`; `rs`/`cs` are the row and column strides in elements.
/// One of each input's strides must be 1 and C must have unit column stride.
///
/// # Safety
/// The pointers must cover the strided matrices and C must not alias A or B.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm(m: usize, k: usize, n: usize, alpha: f32, a: *const f32, rsa: usize, csa: usize, b: *const f32, rsb: usize, csb: usize,
                    beta: f32, c: *mut f32, rsc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    {
        use std::os::raw::c_int;
        let (trans_a, lda) = if csa == 1 { (blas::NO_TRANS, rsa.max(k)) } else { (blas::TRANS, csa.max(m)) };
        let (trans_b, ldb) = if csb == 1 { (blas::NO_TRANS, rsb.max(n)) } else { (blas::TRANS, csb.max(k)) };
        debug_assert!((csa == 1 || rsa == 1) && (csb == 1 || rsb == 1));
        blas::cblas_sgemm(blas::ROW_MAJOR, trans_a, trans_b, m as c_int, n as c_int, k as c_int, alpha, a, lda as c_int, b, ldb as c_int,
                          beta, c, rsc.max(n) as c_int);
    }
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    {
        matrixmultiply::sgemm(m, k, n, alpha, a, rsa as isize, csa as isize, b, rsb as isize, csb as isize, beta, c, rsc as isize, 1);
    }
}

/// Rows per parallel task so that each task has enough work to pay for itself.
pub fn row_chunk(rows: usize, work_per_row: usize) -> usize {
    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return rows.max(1);
    }
    let minimum = (200_000 / work_per_row.max(1)).max(1);
    rows.div_ceil(threads).max(minimum)
}

pub struct Linear {
    pub weight: Vec<f32>, // [outputs, inputs]
    pub bias: Vec<f32>,
    pub inputs: usize,
    pub outputs: usize,
}

impl Linear {
    /// `x` is `[rows, inputs]`; returns `[rows, outputs]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.inputs;
        let mut y = vec![0.0f32; rows * self.outputs];
        self.forward_into(x, &mut y);
        y
    }

    pub fn forward_into(&self, x: &[f32], y: &mut [f32]) {
        let (inputs, outputs) = (self.inputs, self.outputs);
        let rows = x.len() / inputs;
        debug_assert_eq!(y.len(), rows * outputs);
        let chunk = row_chunk(rows, inputs * outputs);
        let run = |x: &[f32], y: &mut [f32]| {
            let rows = x.len() / inputs;
            for row in y.chunks_exact_mut(outputs) {
                row.copy_from_slice(&self.bias);
            }
            // B = weight transposed: element (k, j) is weight[j * inputs + k].
            unsafe { sgemm(rows, inputs, outputs, 1.0, x.as_ptr(), inputs, 1, self.weight.as_ptr(), 1, inputs, 1.0, y.as_mut_ptr(), outputs) };
        };
        if chunk >= rows {
            run(x, y);
        } else {
            y.par_chunks_mut(chunk * outputs).zip(x.par_chunks(chunk * inputs)).for_each(|(y, x)| run(x, y));
        }
    }
}

pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LayerNorm {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        self.apply(&mut y);
        y
    }

    pub fn apply(&self, x: &mut [f32]) {
        let width = self.weight.len();
        let chunk = row_chunk(x.len() / width, 4 * width);
        if chunk * width >= x.len() {
            self.rows(x);
        } else {
            x.par_chunks_mut(chunk * width).for_each(|part| self.rows(part));
        }
    }

    fn rows(&self, x: &mut [f32]) {
        let width = self.weight.len();
        for row in x.chunks_exact_mut(width) {
            let mean = crate::fastmath::sum(row) / width as f32;
            let mut lanes = [0.0f32; 16];
            let chunks = row.chunks_exact(16);
            let rest = chunks.remainder();
            for chunk in chunks {
                for i in 0..16 {
                    lanes[i] += (chunk[i] - mean) * (chunk[i] - mean);
                }
            }
            let variance = (lanes.iter().sum::<f32>() + rest.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()) / width as f32;
            let scale = 1.0 / (variance + 1e-5).sqrt();
            for ((value, weight), bias) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
                *value = (*value - mean) * scale * weight + bias;
            }
        }
    }
}

pub fn gelu(x: &mut [f32]) {
    let chunk = row_chunk(x.len(), 16);
    if chunk >= x.len() {
        x.iter_mut().for_each(|v| *v = crate::fastmath::gelu(*v));
    } else {
        x.par_chunks_mut(chunk).for_each(|part| part.iter_mut().for_each(|v| *v = crate::fastmath::gelu(*v)));
    }
}

/// Unnormalized softmax over each row of `[rows, stride]` in place, where
/// only the first `width` columns hold scores and `stride` is `width` rounded
/// up to a multiple of 16 (so every loop runs on whole vectors). Each row
/// becomes `exp(x - max)` with zeros in the padding; returns 1 / (row sum) per
/// row. The caller applies the scale wherever it is cheapest.
pub fn softmax_rows(x: &mut [f32], width: usize, stride: usize, scales: &mut Vec<f32>) {
    debug_assert!(stride % 16 == 0 && stride >= width);
    scales.clear();
    for row in x.chunks_exact_mut(stride) {
        row[width..].fill(f32::NEG_INFINITY);
        let max = crate::fastmath::max(row);
        for chunk in row.chunks_exact_mut(16) {
            for value in chunk {
                *value = crate::fastmath::exp_nonpositive(*value - max);
            }
        }
        row[width..].fill(0.0);
        scales.push(1.0 / crate::fastmath::sum(row));
    }
}

/// Depthwise 1-D convolution over time with zero padding. `x` is
/// `[frames, channels]`; `weight` is `[kernel, channels]` (tap-major).
pub fn depthwise(x: &[f32], weight: &[f32], bias: &[f32], kernel: usize) -> Vec<f32> {
    let channels = bias.len();
    let frames = x.len() / channels;
    let half = kernel / 2;
    let mut y = vec![0.0f32; x.len()];
    let chunk = row_chunk(frames, channels * kernel);
    let run = |first: usize, out: &mut [f32]| {
        for (offset, row) in out.chunks_exact_mut(channels).enumerate() {
            let t = first + offset;
            row.copy_from_slice(bias);
            for tap in 0..kernel {
                let Some(source) = (t + tap).checked_sub(half).filter(|&s| s < frames) else { continue };
                let input = &x[source * channels..(source + 1) * channels];
                let taps = &weight[tap * channels..(tap + 1) * channels];
                for ((value, input), tap) in row.iter_mut().zip(input).zip(taps) {
                    *value += input * tap;
                }
            }
        }
    };
    if chunk >= frames {
        run(0, &mut y);
    } else {
        y.par_chunks_mut(chunk * channels).enumerate().for_each(|(index, out)| run(index * chunk, out));
    }
    y
}

/// Dense 1-D convolution with zero padding. `x` is `[frames, inputs]`;
/// `taps[k]` is the `[outputs, inputs]` matrix of kernel position `k`.
pub fn conv1d(x: &[f32], taps: &[Vec<f32>], bias: &[f32], inputs: usize) -> Vec<f32> {
    let outputs = bias.len();
    let frames = x.len() / inputs;
    let half = taps.len() / 2;
    let mut y = vec![0.0f32; frames * outputs];
    for row in y.chunks_exact_mut(outputs) {
        row.copy_from_slice(bias);
    }
    for (tap, weight) in taps.iter().enumerate() {
        // Output frame t reads input frame t + tap - half.
        let (first_out, first_in) = if tap >= half { (0, tap - half) } else { (half - tap, 0) };
        if first_in >= frames || first_out >= frames {
            continue;
        }
        let rows = frames - first_in.max(first_out);
        unsafe {
            sgemm(rows, inputs, outputs, 1.0, x.as_ptr().add(first_in * inputs), inputs, 1, weight.as_ptr(), 1, inputs, 1.0,
                  y.as_mut_ptr().add(first_out * outputs), outputs);
        }
    }
    y
}
