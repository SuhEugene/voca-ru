//! Branch-free single-precision `exp`, `ln`, `sin`/`cos` that the compiler
//! vectorizes when they are mapped over slices. Accurate to about 1e-7.

const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: adding and subtracting rounds to nearest

/// Round to the nearest integer (ties to even) for |x| < 2^22.
#[inline(always)]
pub fn round(x: f32) -> f32 {
    (x + ROUND) - ROUND
}

/// Round to the nearest integer for |x| < 2^51.
#[inline(always)]
pub fn round64(x: f64) -> f64 {
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    (x + MAGIC) - MAGIC
}

/// `a * b + c`, fused where the target always has the instruction.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.max(-87.0).min(88.0);
    let n = round(x * std::f32::consts::LOG2_E);
    let r = fma(n, 2.121_944_4e-4, fma(n, -0.693_359_4, x));
    let mut p = 1.987_569_1e-4f32;
    p = fma(p, r, 1.398_199_9e-3);
    p = fma(p, r, 8.333_452e-3);
    p = fma(p, r, 4.166_579_6e-2);
    p = fma(p, r, 1.666_666_5e-1);
    p = fma(p, r, 5.000_000_1e-1);
    let value = fma(p * r, r, r) + 1.0;
    // n is integral and within [-126, 127] after the limits above.
    value * f32::from_bits(((unsafe { n.to_int_unchecked::<i32>() } + 127) as u32) << 23)
}

/// `exp` for arguments known to be at most 0 (softmax, Gaussian tails).
#[inline(always)]
pub fn exp_nonpositive(x: f32) -> f32 {
    let x = x.max(-87.0);
    let n = round(x * std::f32::consts::LOG2_E);
    let r = fma(n, 2.121_944_4e-4, fma(n, -0.693_359_4, x));
    let mut p = 1.987_569_1e-4f32;
    p = fma(p, r, 1.398_199_9e-3);
    p = fma(p, r, 8.333_452e-3);
    p = fma(p, r, 4.166_579_6e-2);
    p = fma(p, r, 1.666_666_5e-1);
    p = fma(p, r, 5.000_000_1e-1);
    let value = fma(p * r, r, r) + 1.0;
    value * f32::from_bits(((unsafe { n.to_int_unchecked::<i32>() } + 127) as u32) << 23)
}

/// Natural logarithm of a positive normal number.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) as i32 - 126) as f32;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000); // [0.5, 1)
    let low = m < std::f32::consts::FRAC_1_SQRT_2;
    e -= if low { 1.0 } else { 0.0 };
    m = if low { m + m - 1.0 } else { m - 1.0 };
    let z = m * m;
    let mut p = 7.037_683_6e-2f32;
    p = fma(p, m, -1.151_461e-1);
    p = fma(p, m, 1.167_699_9e-1);
    p = fma(p, m, -1.242_014_1e-1);
    p = fma(p, m, 1.424_932_3e-1);
    p = fma(p, m, -1.666_805_8e-1);
    p = fma(p, m, 2.000_071_5e-1);
    p = fma(p, m, -2.499_999_4e-1);
    p = fma(p, m, 3.333_333_1e-1);
    let y = fma(e, -2.121_944_4e-4, p * m * z) - 0.5 * z;
    fma(e, 0.693_359_4, m + y)
}

/// `(sin, cos)` of an angle given in turns (1 turn = 2 pi), |turns| < 2^20.
#[inline(always)]
pub fn sin_cos_turns(turns: f32) -> (f32, f32) {
    let k = round(2.0 * turns); // nearest half turn
    let x = fma(k, -0.5, turns) * (2.0 * std::f32::consts::PI); // [-pi/2, pi/2]
    let sign = 1.0 - 2.0 * ((unsafe { k.to_int_unchecked::<i32>() }) & 1) as f32;
    let x2 = x * x;
    let mut s = -2.505_210_8e-8f32;
    s = fma(s, x2, 2.755_731_9e-6);
    s = fma(s, x2, -1.984_127e-4);
    s = fma(s, x2, 8.333_333_3e-3);
    s = fma(s, x2, -1.666_666_7e-1);
    let sin = fma(x * x2, s, x);
    let mut c = 2.087_675_7e-9f32;
    c = fma(c, x2, -2.755_731_9e-7);
    c = fma(c, x2, 2.480_158_7e-5);
    c = fma(c, x2, -1.388_888_9e-3);
    c = fma(c, x2, 4.166_666_8e-2);
    c = fma(c, x2, -0.5);
    let cos = fma(x2, c, 1.0);
    (sign * sin, sign * cos)
}

/// Exact GELU, `x * Phi(x)`, with `erf` from Abramowitz & Stegun 7.1.26
/// (absolute error below 1.5e-7).
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / fma(0.327_591_1, z, 1.0);
    let mut p = 1.061_405_4f32;
    p = fma(p, t, -1.453_152_1);
    p = fma(p, t, 1.421_413_8);
    p = fma(p, t, -0.284_496_72);
    p = fma(p, t, 0.254_829_6);
    let tail = 0.5 * p * t * exp_nonpositive(-z * z); // 0.5 * erfc(|x| / sqrt 2) = Phi(-|x|)
    x * if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Sum with independent partial sums so the loop vectorizes.
#[inline]
pub fn sum(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 16];
    let chunks = x.chunks_exact(16);
    let rest = chunks.remainder();
    for chunk in chunks {
        for i in 0..16 {
            lanes[i] += chunk[i];
        }
    }
    lanes.iter().sum::<f32>() + rest.iter().sum::<f32>()
}

#[inline]
pub fn max(x: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; 16];
    let chunks = x.chunks_exact(16);
    let rest = chunks.remainder();
    for chunk in chunks {
        for i in 0..16 {
            lanes[i] = if chunk[i] > lanes[i] { chunk[i] } else { lanes[i] };
        }
    }
    lanes.iter().chain(rest).fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_std() {
        for i in -8000..8000 {
            let x = i as f32 * 0.01;
            assert!((exp(x) / x.exp() - 1.0).abs() < 4e-7, "exp {x}");
            let (s, c) = sin_cos_turns(x);
            let angle = x as f64 * 2.0 * std::f64::consts::PI;
            assert!((s as f64 - angle.sin()).abs() < 2e-5 && (c as f64 - angle.cos()).abs() < 2e-5, "sincos {x}");
        }
        for i in -2500..2500 {
            let x = i as f32 * 1e-4;
            let (s, c) = sin_cos_turns(x);
            let angle = x as f64 * 2.0 * std::f64::consts::PI;
            assert!((s as f64 - angle.sin()).abs() < 3e-7 && (c as f64 - angle.cos()).abs() < 3e-7, "sincos {x}");
        }
        for i in 1..20000 {
            let x = i as f32 * 0.37e-3;
            assert!((ln(x) - x.ln()).abs() < 3e-7 * x.ln().abs().max(1.0), "ln {x}");
        }
        assert!((ln(2f32.powi(-24)) - 2f32.powi(-24).ln()).abs() < 1e-5);
    }
}
