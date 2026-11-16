//! Minimal safetensors reader (little-endian F32 / I32 tensors).

use std::collections::HashMap;
use std::path::Path;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub f32: Vec<f32>,
    pub i32: Vec<i32>,
}

pub struct Archive {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("not a safetensors file".into());
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let body = 8usize.checked_add(header_len).filter(|&end| end <= bytes.len()).ok_or("truncated header")?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..body]).map_err(|e| e.to_string())?;
        let data = &bytes[body..];
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, entry) in header.as_object().ok_or("header is not an object")? {
            if name == "__metadata__" {
                for (key, value) in entry.as_object().into_iter().flatten() {
                    metadata.insert(key.clone(), value.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let shape: Vec<usize> = entry["shape"].as_array().ok_or("missing shape")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
            let start = entry["data_offsets"][0].as_u64().ok_or("missing offsets")? as usize;
            let end = entry["data_offsets"][1].as_u64().ok_or("missing offsets")? as usize;
            let raw = data.get(start..end).ok_or("tensor data out of range")?;
            let count: usize = shape.iter().product();
            if raw.len() != 4 * count {
                return Err(format!("{name}: size does not match shape"));
            }
            let words = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]);
            let mut tensor = Tensor { shape, f32: Vec::new(), i32: Vec::new() };
            match entry["dtype"].as_str() {
                Some("F32") => tensor.f32 = words.map(f32::from_le_bytes).collect(),
                Some("I32") => tensor.i32 = words.map(i32::from_le_bytes).collect(),
                other => return Err(format!("{name}: unsupported dtype {other:?}")),
            }
            tensors.insert(name.clone(), tensor);
        }
        Ok(Self { tensors, metadata })
    }

    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let tensor = self.tensors.remove(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if tensor.shape != shape {
            return Err(format!("{name}: shape {:?}, expected {:?}", tensor.shape, shape));
        }
        Ok(tensor.f32)
    }
}
