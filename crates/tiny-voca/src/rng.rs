//! Random numbers that reproduce `torch.randn(n, generator=manual_seed(s))` on
//! the CPU: MT19937, 24-bit uniforms and the 16-wide Box-Muller of PyTorch.

use crate::fastmath::{ln, sin_cos_turns};

pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn refill(&mut self) {
        #[inline(always)]
        fn twist(current: u32, next: u32, far: u32) -> u32 {
            let y = (current & 0x8000_0000) | (next & 0x7fff_ffff);
            far ^ (y >> 1) ^ ((y & 1).wrapping_neg() & 0x9908_b0df)
        }
        let s = &mut self.state;
        for i in 0..227 {
            s[i] = twist(s[i], s[i + 1], s[i + 397]);
        }
        for i in 227..623 {
            s[i] = twist(s[i], s[i + 1], s[i - 227]);
        }
        s[623] = twist(s[623], s[0], s[396]);
        self.index = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.refill();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u32() & ((1 << 24) - 1)) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// Standard normal samples from the same random stream and the same
/// Box-Muller pairing as `torch.randn(count, generator=manual_seed(seed))`;
/// values agree with PyTorch's to float rounding (`count >= 16`).
pub fn randn(count: usize, seed: u64) -> Vec<f32> {
    let mut generator = Mt19937::new(seed);
    let mut data: Vec<f32> = (0..count).map(|_| generator.uniform()).collect();
    if count < 16 {
        // PyTorch draws short tensors one double at a time; never needed here
        // (an utterance has at least one frame of 256 samples).
        let mut block = [0.5f32; 16];
        block[..count].copy_from_slice(&data);
        box_muller(&mut block);
        data.copy_from_slice(&block[..count]);
        return data;
    }
    for block in data.chunks_exact_mut(16) {
        box_muller(block.try_into().unwrap());
    }
    if count % 16 != 0 {
        let tail: &mut [f32; 16] = (&mut data[count - 16..]).try_into().unwrap();
        for value in tail.iter_mut() {
            *value = generator.uniform();
        }
        box_muller(tail);
    }
    data
}

fn box_muller(block: &mut [f32; 16]) {
    let (low, high) = block.split_at_mut(8);
    for j in 0..8 {
        let radius = (-2.0 * ln(1.0 - low[j])).sqrt();
        let (sin, cos) = sin_cos_turns(high[j]); // theta = 2 pi u2
        low[j] = radius * cos;
        high[j] = radius * sin;
    }
}
