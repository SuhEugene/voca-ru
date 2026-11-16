//! WAV output: 32-bit float (what the reference writes) or 16-bit PCM.

use std::io::Write;
use std::path::Path;

pub fn write(path: &Path, samples: &[f32], sample_rate: u32, pcm16: bool) -> std::io::Result<()> {
    let (format, bytes_per_sample) = if pcm16 { (1u16, 2u32) } else { (3u16, 4u32) };
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes());
    out.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes());
    out.extend_from_slice(&(bytes_per_sample as u16 * 8).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        if pcm16 {
            out.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        } else {
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::create(path)?.write_all(&out)
}
